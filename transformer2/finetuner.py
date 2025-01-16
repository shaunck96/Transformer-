#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer² Production-Grade Example
-------------------------------------
Demonstrates a more robust approach with:
 - Accelerate for distributed training
 - PPO-like RL stable training (via 'trl')
 - Advanced logging and checkpointing
 - SVD-based Fine-Tuning (SVF) with stable merges
 - Task adaptation with prompt-based, classifier-based, or few-shot methods
   (including a more robust CEM for few-shot search)
 - A mock "DeepSeek Instruct Model" from Hugging Face as the base
"""

import os
import sys
import math
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.stats import truncnorm

# Hugging Face & RL imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from accelerate import Accelerator
from trl import PPOTrainer, PPOConfig  # A simplified interface for PPO

# --------------------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("transformer2")


# --------------------------------------------------------------------------------
# SVD Utilities
# --------------------------------------------------------------------------------
class SvdComponent:
    """
    Encapsulates the SVD decomposition for a single parameter matrix W.
    """
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor):
        self.U = U
        self.S = S
        self.Vt = Vt

    def device(self):
        return self.U.device


def decompose_param_matrix(param: torch.nn.Parameter, full_matrices: bool = False) -> SvdComponent:
    """
    Decompose a parameter matrix into (U, S, V^T).
    """
    with torch.no_grad():
        U, S, Vt = torch.linalg.svd(param.data, full_matrices=full_matrices)
    return SvdComponent(U, S, Vt)


def reconstruct_weight(svd_comp: SvdComponent, z: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct W' = U diag(S * z) V^T.
    """
    scaled_s = svd_comp.S * z
    diag_scaled_s = torch.diag(scaled_s)
    W_prime = svd_comp.U @ diag_scaled_s @ svd_comp.Vt
    return W_prime


# --------------------------------------------------------------------------------
# SVF-Wrapper for Fine-tuning
# --------------------------------------------------------------------------------
class SVFWrapper(nn.Module):
    """
    A module that holds:
      - The original pretrained model (with *frozen* params).
      - The SVD decompositions for selected parameters.
      - The learned z-vectors as trainable parameters.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        svd_map: Dict[str, SvdComponent],
        init_mean: float = 0.05,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.base_model = base_model
        self.svd_map = svd_map

        # We freeze base_model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Create parameter dict for z
        self.z_params = nn.ParameterDict()
        for pname, svd_comp in self.svd_map.items():
            rank_r = svd_comp.S.shape[0]
            # Truncated normal or small uniform
            init_z = truncnorm.rvs(-2, 2, loc=init_mean, scale=init_std, size=rank_r)
            z_t = torch.tensor(init_z, dtype=torch.float32)
            self.z_params[pname] = nn.Parameter(z_t)

    def forward(self, *args, **kwargs):
        """
        We typically do not use forward() directly. Instead, we do 
        weight patching + base_model's forward. This is a partial stub.
        """
        return self.base_model(*args, **kwargs)

    @torch.no_grad()
    def patch_weights(self) -> None:
        """
        Reconstruct the weights from SVD and the current z-params.
        This method modifies the base_model in-place.
        """
        for pname, z_vec in self.z_params.items():
            if pname not in self.svd_map:
                continue
            svd_comp = self.svd_map[pname]
            new_w = reconstruct_weight(svd_comp, z_vec)
            # Now we find the base model parameter with the same name and copy
            param_ref = dict(self.base_model.named_parameters())[pname]
            param_ref.data.copy_(new_w)

    def named_zparams(self):
        """
        Return the name & param for the trainable z-parameters
        """
        for n, p in self.z_params.items():
            yield n, p

    def parameters_for_optimization(self):
        """
        Return only z-params as a list for optimization
        """
        return list(self.z_params.values())


# --------------------------------------------------------------------------------
# RL Fine-tuning with PPO (Simplified)
# --------------------------------------------------------------------------------
class SVF_PPOTrainer:
    """
    High-level PPO trainer using the `trl` library from Hugging Face.
    We only train the z-parameters, not the entire base model.
    """

    def __init__(
        self,
        svf_wrapper: SVFWrapper,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
        ppo_config: PPOConfig,
    ):
        self.accelerator = accelerator
        self.svf_wrapper = svf_wrapper
        self.tokenizer = tokenizer

        # Patch weights once before starting
        self.svf_wrapper.patch_weights()

        # Build the PPO trainer from `trl`
        # But we only want to optimize self.svf_wrapper's z-params
        # We'll do a trick: create a "dummy" reference model for KL
        self.ref_model = self._build_reference_model()
        self.ref_model.eval()

        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.svf_wrapper.base_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=None,   # we'll feed data on the fly
            data_collator=None,
        )

        # Adjust the PPO optimizer to only optimize z-params
        # This requires a small hack: set requires_grad for base_model false, except the patched ones.
        # In practice, we can replace the ppo_trainer's optimizer with a custom one:
        opt_params = [{"params": self.svf_wrapper.parameters_for_optimization(), "lr": ppo_config.lr}]
        self.ppo_trainer.optimizer = torch.optim.AdamW(opt_params)

    def _build_reference_model(self) -> PreTrainedModel:
        # Make a deep copy of the base model, 
        # but we keep it with the same architecture for KL references.
        import copy
        ref_model = copy.deepcopy(self.svf_wrapper.base_model)
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    @torch.no_grad()
    def generate(self, prompt: str, max_length=128) -> str:
        """Generate text from the patched model."""
        self.svf_wrapper.patch_weights()  # Ensure patched weights
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.accelerator.device)
        output_ids = self.svf_wrapper.base_model.generate(
            **inputs, 
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.9
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def train_on_prompts(
        self,
        prompts: List[str],
        target_texts: List[str],
        batch_size: int = 4,
        epochs: int = 1
    ):
        """
        Example PPO training loop. 
        `target_texts` is used for simplistic reward shaping: +1 if substring present, else 0.
        This part depends heavily on your application.
        """
        dataset = list(zip(prompts, target_texts))
        steps_per_epoch = math.ceil(len(dataset) / batch_size)

        for epoch in range(epochs):
            logger.info(f"Starting PPO Epoch [{epoch+1}/{epochs}]")
            np.random.shuffle(dataset)
            for step_i in range(steps_per_epoch):
                batch = dataset[step_i * batch_size : (step_i + 1) * batch_size]
                if not batch:
                    continue

                # Patch weights once in the loop
                self.svf_wrapper.patch_weights()

                # Gather samples
                query_tensors = []
                response_tensors = []
                rewards = []

                for prompt, target in batch:
                    prompt_enc = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.accelerator.device)
                    # Generate
                    response_ids = self.svf_wrapper.base_model.generate(
                        prompt_enc, 
                        max_new_tokens=64,
                        do_sample=True,
                        temperature=0.7
                    )
                    # Convert to string
                    response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
                    
                    # Reward shaping
                    # +1 if target is substring, else 0
                    r = 1.0 if target.strip() in response_text else 0.0

                    query_tensors.append(prompt_enc[0])
                    response_tensors.append(response_ids[0])
                    rewards.append(torch.tensor(r, dtype=torch.float32, device=self.accelerator.device))

                # PPO step: 
                # Note: the next step is typically handled in `self.ppo_trainer.step`:
                self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Optionally: log or checkpoint
            # e.g. self.save_z_checkpoint(f"svf_ppo_epoch_{epoch}.pt")
            logger.info(f"Finished PPO Epoch [{epoch+1}/{epochs}]")

    def save_z_checkpoint(self, path: str):
        """
        Saves the current z-parameters to disk.
        """
        logger.info(f"Saving z-params to {path}")
        # gather z-params from self.svf_wrapper
        z_state = {}
        for pname, param in self.svf_wrapper.z_params.items():
            z_state[pname] = param.detach().cpu().numpy()
        with open(path, 'wb') as f:
            torch.save(z_state, f)

    def load_z_checkpoint(self, path: str):
        """
        Loads the z-parameters from disk, sets them in the model, patches.
        """
        logger.info(f"Loading z-params from {path}")
        with open(path, 'rb') as f:
            z_state = torch.load(f)
        for pname, arr in z_state.items():
            if pname in self.svf_wrapper.z_params:
                t = torch.from_numpy(arr).to(self.accelerator.device)
                self.svf_wrapper.z_params[pname].data.copy_(t)
        self.svf_wrapper.patch_weights()


# --------------------------------------------------------------------------------
# Adaptation Methods
# --------------------------------------------------------------------------------

def classify_prompt(llm_generate_fn, question: str) -> str:
    """
    Use an LLM-based approach to classify the question into: 'math', 'code', 'reasoning', or 'others'.
    Returns the classification as a string.
    """
    prompt = (
        "Please classify the following question into one of the categories:\n"
        "'math', 'code', 'reasoning', 'others'.\n\n"
        f"Question: {question}\n"
        "Answer with one category label only."
    )
    gen_text = llm_generate_fn(prompt, max_length=64)
    # Naive parse
    gen_text = gen_text.lower()
    if "math" in gen_text:
        return "math"
    elif "code" in gen_text:
        return "code"
    elif "reasoning" in gen_text:
        return "reasoning"
    else:
        return "others"


def compute_cem_interpolation(
    svf_wrapper_map: Dict[str, SVFWrapper],
    param_names: List[str],
    fewshot_data: List[Tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    base_model: PreTrainedModel,
    accelerator: Accelerator,
    max_iter=30,
    pop_size=20,
    elite_frac=0.2
) -> Dict[str, torch.Tensor]:
    """
    Example CEM approach that finds alpha for each domain to combine multiple experts.
    For demonstration, we do a single global alpha per expert domain.

    Returns a dictionary param_name -> final z vector 
    that is the weighted sum of experts' z vectors.
    """

    domain_labels = list(svf_wrapper_map.keys())  # e.g. ['math', 'code', 'reasoning', 'others']
    K = len(domain_labels)

    # Prepare expert z vectors for each param
    # expert_z_map[ p ][ d ] -> a Tensor of shape [rank_r]
    expert_z_map = {}
    for dlabel, w in svf_wrapper_map.items():
        for pname, zparam in w.z_params.items():
            expert_z_map.setdefault(pname, {})[dlabel] = zparam.detach().clone()

    # We'll search for alpha in R^K
    def gen_alpha(mu, sigma):
        return np.random.normal(mu, sigma, size=K)

    def evaluate_alpha(alpha: np.ndarray) -> float:
        """
        1) For each param, build the new z = sum_k alpha[k] * z^k
        2) Patch the base_model
        3) Evaluate on few-shot data
        """
        new_z_dict = {}
        for pname, domain_zdict in expert_z_map.items():
            # Weighted sum
            z_sum = None
            for i, dlabel in enumerate(domain_labels):
                z_e = domain_zdict[dlabel]
                if z_sum is None:
                    z_sum = alpha[i] * z_e
                else:
                    z_sum += alpha[i] * z_e
            new_z_dict[pname] = z_sum

        # Patch
        original_data = {}
        for pname, param in base_model.named_parameters():
            if pname in param_names:
                original_data[pname] = param.data.clone()

        with torch.no_grad():
            for pname, z_val in new_z_dict.items():
                # Reconstruct
                svd_comp = svf_wrapper_map[domain_labels[0]].svd_map[pname]  # any wrapper has same shape
                w_prime = reconstruct_weight(svd_comp, z_val)
                dict(base_model.named_parameters())[pname].copy_(w_prime)

        # Evaluate correctness
        correct = 0
        for q, ans in fewshot_data:
            inputs = tokenizer(q, return_tensors="pt").to(accelerator.device)
            out_ids = base_model.generate(**inputs, max_new_tokens=40)
            gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            if ans.strip() in gen_text:
                correct += 1

        # restore
        with torch.no_grad():
            for pname, data_orig in original_data.items():
                dict(base_model.named_parameters())[pname].copy_(data_orig)

        return correct / len(fewshot_data)

    # CEM loop
    mu = np.ones(K) / K
    sigma = np.ones(K) * 0.5
    n_elite = int(pop_size * elite_frac)

    best_alpha = mu.copy()
    best_score = 0.0

    for it in range(max_iter):
        samples = [gen_alpha(mu, sigma) for _ in range(pop_size)]
        scores = []
        for s in samples:
            score = evaluate_alpha(s)
            scores.append(score)

        idx_sorted = np.argsort(scores)[::-1]
        elites = [samples[i] for i in idx_sorted[:n_elite]]
        elite_scores = [scores[i] for i in idx_sorted[:n_elite]]
        elites_np = np.array(elites)
        mu = elites_np.mean(axis=0)
        sigma = elites_np.std(axis=0)

        if elite_scores[0] > best_score:
            best_score = elite_scores[0]
            best_alpha = elites_np[0]

    # Build final z
    final_z_dict = {}
    for pname, domain_zdict in expert_z_map.items():
        z_sum = None
        for i, dlabel in enumerate(domain_labels):
            if z_sum is None:
                z_sum = best_alpha[i] * domain_zdict[dlabel]
            else:
                z_sum += best_alpha[i] * domain_zdict[dlabel]
        final_z_dict[pname] = z_sum
    return final_z_dict


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer² Production Example")
    parser.add_argument("--model_name", type=str, default="DeepSeekInstruct/large", help="HF Model ID")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to store checkpoints")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    logger.info(f"Using device: {accelerator.device}, n_processes={accelerator.num_processes}")

    # 1) Load base model & tokenizer
    logger.info(f"Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    param_names_to_svf = []
    for n, p in base_model.named_parameters():
        if "attn" in n or "mlp" in n:
            param_names_to_svf.append(n)

    # 2) Create SVD map
    svd_map = {}
    for pname, p in base_model.named_parameters():
        if pname in param_names_to_svf:
            svd_map[pname] = decompose_param_matrix(p, full_matrices=False)

    # 3) Create an SVF wrapper 
    svf_wrapper = SVFWrapper(base_model, svd_map, init_mean=0.05, init_std=0.01)
    # We'll do PPO-like RL with trl, focusing only on z-params

    # 4) Setup PPO config
    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        forward_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_with=None,  # We can integrate wandb or tensorboard
        optimize_cuda_cache=True,
    )
    # Create PPO trainer
    svf_ppo = SVF_PPOTrainer(svf_wrapper, tokenizer, accelerator, ppo_config)

    # 5) Mock dataset: For math domain
    # (Replace with real data, e.g. from GSM8K or your custom set)
    train_prompts = [
        "What is 2 + 2?",
        "Compute 10 * 3.",
        "Compute 7+5.",
        "Find 100 minus 1.",
    ] * 20
    train_targets = [
        "4",
        "30",
        "12",
        "99",
    ] * 20

    # 6) Distribute with Accelerator
    # Not needed for dataset of small size, but we show usage
    train_prompts, train_targets = accelerator.prepare(train_prompts, train_targets)

    logger.info("Starting PPO-based training on the math domain ...")
    svf_ppo.train_on_prompts(
        prompts=train_prompts,
        target_texts=train_targets,
        batch_size=args.batch_size,
        epochs=args.max_epochs
    )

    # Save the checkpoint of z
    os.makedirs(args.output_dir, exist_ok=True)
    z_ckpt_path = os.path.join(args.output_dir, "svf_zparams_math.pt")
    svf_ppo.save_z_checkpoint(z_ckpt_path)

    # 7) Example: Inference with adaptation
    logger.info("=== Example: Prompt-based adaptation ===")
    question = "Compute 15 + 37."
    # classification phase
    pred_domain = classify_prompt(svf_ppo.generate, question)
    logger.info(f"Classification says domain: {pred_domain}")
    # For demonstration, let's assume we only have "math" vs "others"
    # We'll just load the math z-checkpoint if the domain is "math"
    if pred_domain == "math":
        svf_ppo.load_z_checkpoint(z_ckpt_path)
    else:
        # We might have an "others" checkpoint or do no adaptation
        pass

    # Now generate the final
    answer = svf_ppo.generate(question)
    logger.info(f"Final answer after prompt-based adaptation: {answer}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
