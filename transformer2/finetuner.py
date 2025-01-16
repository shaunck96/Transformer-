#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example End-to-End Implementation of Transformer²: Self-Adaptive LLMs

Author: Your Name
Date: 2025-01-16
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# For demonstration with a small model
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

############################################################
# SVD-based Fine-tuning (SVF)
############################################################

@dataclass
class SvdComponent:
    """
    Dataclass that encapsulates relevant SVD decomposition info
    for a single parameter matrix W.
    """
    U: torch.Tensor      # Shape: (m, r)
    S: torch.Tensor      # Shape: (r,)
    Vt: torch.Tensor     # Shape: (r, n)
    shape_m: int
    shape_n: int

def compute_svd_matrices(model: nn.Module, param_names: List[str]) -> Dict[str, SvdComponent]:
    """
    Decompose the selected model weight matrices into their SVD components.

    Args:
        model: The LLM or smaller model with weight matrices.
        param_names: A list of parameter names for which we want to do SVD.

    Returns:
        A dict mapping parameter name -> SvdComponent.
    """
    svd_dict = {}
    for name, param in model.named_parameters():
        if name in param_names:
            W = param.data
            # W shape: (m, n)
            m, n = W.shape
            # Full SVD or truncated SVD. Full SVD for demonstration:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            # Store
            svd_dict[name] = SvdComponent(
                U=U, 
                S=S, 
                Vt=Vt, 
                shape_m=m, 
                shape_n=n
            )
    return svd_dict

def assemble_weight_from_svd(svd_comp: SvdComponent, z: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct the weight matrix W' = U (S * z) V^T.

    Args:
        svd_comp: Precomputed SVD components for a parameter
        z: The scaling vector z for the singular values, shape (r,)

    Returns:
        The reconstructed weight W'.
    """
    # shape mismatch checks omitted for brevity
    scaled_s = svd_comp.S * z
    # Diagonalize scaled_s => (r, r)
    diag_scaled_s = torch.diag(scaled_s)
    # Reconstruct
    W_prime = svd_comp.U @ diag_scaled_s @ svd_comp.Vt
    return W_prime

class SVFModelWrapper(nn.Module):
    """
    Wrapper that takes a pretrained model + a dictionary of SVD components,
    and maintains a set of learnable z-vectors (one per parameter).
    
    The forward pass reconstructs the relevant weight matrices on the fly.
    In practice for efficiency, you might reconstruct these once per update,
    or do partial reconstruction. This code is simplified for clarity.
    """

    def __init__(self, 
                 base_model: nn.Module, 
                 svd_dict: Dict[str, SvdComponent], 
                 device: str = "cpu"):
        super().__init__()
        self.base_model = base_model
        self.svd_dict = svd_dict
        
        # We'll keep a learnable z for each param in self.svd_dict
        self.z_params = nn.ParameterDict()

        for pname, svd_comp in self.svd_dict.items():
            r = svd_comp.S.shape[0]
            # Initialize around 0.1 (as recommended in the paper)
            z = torch.ones(r, device=device) * 0.1
            self.z_params[pname] = nn.Parameter(z)
        
        self.device_ = device
        self.to(device)

    def forward(self, *args, **kwargs):
        # 1) Reconstruct the relevant weight from SVD + z
        # 2) Temporarily patch it into self.base_model
        # 3) Forward pass with the patched weights
        # 
        # Because we want to avoid leaving the model in a partial state,
        # we'll store original parameters, patch them, then restore them.

        original_params = {}
        for pname, param in self.base_model.named_parameters():
            if pname in self.svd_dict:
                original_params[pname] = param.data

        # Patch
        for pname, svd_comp in self.svd_dict.items():
            param_data = assemble_weight_from_svd(
                svd_comp, self.z_params[pname]
            )
            # set the param in the base model
            with torch.no_grad():
                target_param = dict(self.base_model.named_parameters())[pname]
                target_param.copy_(param_data)

        # Forward
        outputs = self.base_model(*args, **kwargs)

        # Restore
        for pname, saved_data in original_params.items():
            with torch.no_grad():
                dict(self.base_model.named_parameters())[pname].copy_(saved_data)

        return outputs

    def get_policy_parameters(self):
        """
        Get a list of only the z-parameters for optimization (RL or others).
        """
        return list(self.z_params.values())

############################################################
# RL Training Stub
############################################################

class SimpleRLTrainer:
    """
    A minimal policy-gradient trainer to fine-tune the z-vectors.

    We assume that the environment returns reward = +1 if correct, -1 if incorrect,
    or 0 otherwise. This is a toy-like approach for demonstration only.
    """

    def __init__(self, 
                 svf_model: SVFModelWrapper, 
                 tokenizer, 
                 lr: float = 2e-3, 
                 kl_coeff: float = 0.0, 
                 device: str = "cpu"):
        self.model = svf_model
        self.tokenizer = tokenizer
        self.optimizer = optim.AdamW(svf_model.get_policy_parameters(), lr=lr)
        self.kl_coeff = kl_coeff
        self.device = device

        # We store reference to base model for KL penalty
        self.ref_model = copy.deepcopy(svf_model.base_model).eval().requires_grad_(False)

    def generate(self, prompt: str, max_new_tokens=64) -> str:
        """
        Generate a response from the model for a given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def train_on_dataset(
        self,
        dataset: List[Tuple[str, str]],
        batch_size: int = 32, 
        epochs: int = 5
    ):
        """
        dataset: List of (question, reference_answer)
        The reference answer is used purely to check correctness for RL reward.
        """
        data_size = len(dataset)
        for epoch in range(epochs):
            random.shuffle(dataset)
            epoch_losses = []
            for start_idx in range(0, data_size, batch_size):
                batch_data = dataset[start_idx : start_idx + batch_size]
                # We'll do naive REINFORCE: sample an answer, compute reward,
                # compute logprob (with forward pass).
                # This is a simplified approach; you can do better with
                # advanced RL frameworks or libraries.
                self.optimizer.zero_grad()

                log_probs = []
                rewards = []
                for (question, ref_answer) in batch_data:
                    # Build input
                    inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
                    # Forward pass to get logprobs
                    # We'll do a naive approach: compute logprobs for next token
                    # or the entire sequence. This is not perfect but for brevity.

                    # Reconstruct patched weights
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    # outputs.logits shape [batch_size, seq_len, vocab_size]
                    # negative log-likelihood
                    # For demonstration: use average over tokens
                    nll = outputs.loss  
                    # Compute "log prob" as -nll
                    log_p = -nll

                    # Generate for reward
                    gen_text = self.generate(question)
                    # Reward stub: +1 if substring match, else -1
                    # In your real code, parse carefully or use specialized check
                    if ref_answer.strip() in gen_text:
                        r = 1.0
                    else:
                        r = -1.0

                    rewards.append(r)
                    log_probs.append(log_p)

                # Combine
                log_probs_t = torch.stack(log_probs)
                rewards_t = torch.tensor(rewards, dtype=torch.float, device=self.device)
                # REINFORCE gradient => - E[r * log_probs]
                # We'll define a policy loss:
                loss = - (rewards_t * log_probs_t).mean()

                # Add KL penalty
                if self.kl_coeff > 0.0:
                    # measure distance to ref_model outputs
                    # Do a quick pass on the same batch for the reference
                    # This is extremely naive, but for demonstration
                    kl_losses = []
                    for (question, _) in batch_data:
                        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            ref_outputs = self.ref_model(**inputs, labels=inputs["input_ids"])
                        # kl = difference in logits? We'll do a quick difference in losses
                        # again naive for demonstration
                        current_outputs = self.model(**inputs, labels=inputs["input_ids"])
                        kl_val = current_outputs.loss - ref_outputs.loss
                        kl_losses.append(kl_val)
                    kl_val_t = torch.stack(kl_losses).mean()
                    loss = loss + self.kl_coeff * kl_val_t

                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            avg_epoch_loss = np.mean(epoch_losses)
            print(f"[Epoch {epoch+1}/{epochs}] avg_loss = {avg_epoch_loss:.4f}")

############################################################
# Adaptation Methods
############################################################

def prompt_based_adaptation_llm(question: str) -> str:
    """
    Minimal prompt used to classify the question into one of the known categories: 'math', 'code', 'reasoning', 'others'.
    """
    # In production, you'd load a system or chain-of-thought approach.  
    prompt = f"""Analyze the given question and classify it into one of the following:
  'math', 'code', 'reasoning', or 'others'. 
  The question is: "{question}"
  Provide your final classification enclosed in \\boxed{{}}.
Classification: \\boxed{{"""
    return prompt

def adapt_select_z_vector(
    classification: str, 
    z_vectors_library: Dict[str, torch.Tensor]
) -> Optional[torch.Tensor]:
    """
    Simple mapping from classification -> correct z vector.
    If classification not recognized, returns None.
    """
    classification = classification.strip().lower()
    if classification not in z_vectors_library:
        return None
    return z_vectors_library[classification]

def linear_interpolation_of_z_vectors(
    alpha: List[float], 
    z_vectors: List[torch.Tensor]
) -> torch.Tensor:
    """
    Weighted sum of multiple z vectors of the same dimension.
    alpha and z_vectors must have same length.
    """
    # shape checks omitted
    return sum(a * z for a, z in zip(alpha, z_vectors))

class FewShotAdapter:
    """
    Implementation of the CEM-based search described in the paper 
    for few-shot adaptation by combining multiple learned z vectors.
    """
    def __init__(
        self, 
        base_model: nn.Module, 
        tokenizer, 
        z_vectors_library: Dict[str, torch.Tensor],
        few_shot_data: List[Tuple[str, str]],
        device: str = "cpu"
    ):
        """
        z_vectors_library: e.g. { 'math': z_math, 'code': z_code, 'reasoning': z_reasoning }
        few_shot_data: small data (Q, correct A) used for searching the best alpha
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.z_vectors_library = z_vectors_library
        self.labels = list(z_vectors_library.keys())  # e.g. ['math', 'code', 'reasoning']
        self.device = device
        self.few_shot_data = few_shot_data

    def generate_text(self, prompt: str, W_prime: Dict[str, torch.Tensor]) -> str:
        """
        Patch the base_model with W_prime, then generate.
        For demonstration, we do minimal patching.
        """
        # Save original
        original_params = {}
        for pname, param in self.base_model.named_parameters():
            original_params[pname] = param.data

        # Patch
        with torch.no_grad():
            for pname, new_w in W_prime.items():
                dict(self.base_model.named_parameters())[pname].copy_(new_w)

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out_ids = self.base_model.generate(
            **inputs, 
            max_new_tokens=64
        )
        result = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Restore
        for pname, p_data in original_params.items():
            with torch.no_grad():
                dict(self.base_model.named_parameters())[pname].copy_(p_data)
        return result

    def evaluate_alpha(self, alpha: np.ndarray) -> float:
        """
        Evaluate a candidate alpha (one interpolation weight per z_vector).
        Return a performance measure on the few_shot_data.
        """
        # Build an updated set of param => new W
        # We'll assume each parameter has the same alpha combination,
        # i.e. we do the same interpolation for each param. 
        # (You could do per-layer alpha, which is more complicated.)
        W_prime = {}
        for pname, svd_comp in self.svd_dict.items():
            # gather all relevant z vectors
            candidate_zs = [self.z_vectors_library[label][pname] for label in self.labels]
            # merge
            combined_z = torch.zeros_like(candidate_zs[0])
            for i, z_i in enumerate(candidate_zs):
                combined_z += alpha[i] * z_i
            # reconstruct
            new_w = assemble_weight_from_svd(svd_comp, combined_z)
            W_prime[pname] = new_w

        # Evaluate correctness
        correct_count = 0
        for (q, correct_ans) in self.few_shot_data:
            gen_ans = self.generate_text(q, W_prime)
            if correct_ans.strip() in gen_ans:
                correct_count += 1
        return float(correct_count) / len(self.few_shot_data)

    def cem_search(self, 
                   svd_dict: Dict[str, SvdComponent], 
                   max_iter=30, 
                   pop_size=20, 
                   elite_frac=0.2) -> List[float]:
        """
        Cross-Entropy Method to find alpha that best fits few_shot_data.
        alpha dimension = len(self.labels).
        """
        self.svd_dict = svd_dict  # store for usage
        dim = len(self.labels)
        # init distribution (mean=0.33..., std=1.0)
        mu = np.array([1.0 / dim] * dim)
        sigma = np.ones(dim) * 1.0

        n_elite = int(pop_size * elite_frac)
        best_alpha = mu.copy()
        best_score = 0.0

        for iteration in range(max_iter):
            # sample
            samples = []
            for _ in range(pop_size):
                alpha_candidate = np.random.normal(mu, sigma)
                # we can do either bounding [0,1], or free
                # For demonstration, let's keep them unconstrained
                samples.append(alpha_candidate)
            # evaluate
            scores = []
            for alpha_candidate in samples:
                # convert to float
                alpha_candidate_t = torch.tensor(alpha_candidate, dtype=torch.float)
                # evaluate
                score = self.evaluate_alpha(alpha_candidate_t.numpy())
                scores.append(score)

            # pick elites
            idx_sorted = np.argsort(scores)[::-1]
            elites = [samples[i] for i in idx_sorted[:n_elite]]
            elite_scores = [scores[i] for i in idx_sorted[:n_elite]]
            # update distribution
            elites_np = np.array(elites)
            mu = elites_np.mean(axis=0)
            sigma = elites_np.std(axis=0)
            # track best
            if elite_scores[0] > best_score:
                best_score = elite_scores[0]
                best_alpha = elites_np[0]
            
        return best_alpha.tolist()

############################################################
# Main Orchestration
############################################################

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Initialize a base model (for demonstration: GPT2)
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Suppose we want to do SVF on certain modules only (e.g., MLP or attention). 
    # For simplicity, let's pick all "attn.c_attn.weight" in the GPT2's attention layers. 
    # Adjust your param selection as you see fit.
    param_names = []
    for name, _ in base_model.named_parameters():
        if "attn.c_attn.weight" in name:
            param_names.append(name)

    # 2) Compute SVD for those parameters
    print("Computing SVD for selected params ...")
    svd_dict = compute_svd_matrices(base_model, param_names)
    print(f"Found {len(svd_dict)} parameters to SVF tune.")

    # 3) Build an SVF wrapper
    svf_wrapper = SVFModelWrapper(
        base_model=base_model,
        svd_dict=svd_dict,
        device=device
    )

    # 4) RL trainer (purely demonstration)
    rl_trainer = SimpleRLTrainer(
        svf_model=svf_wrapper,
        tokenizer=tokenizer,
        lr=2e-3,
        kl_coeff=0.1,
        device=device
    )

    # 5) Some small synthetic dataset to illustrate
    # dataset of form (question, correct_answer_substring)
    training_data_math = [
        ("What is 2+2?", "4"),
        ("Compute 10*5?", "50"),
        ("Compute 7+9?", "16"),
    ] * 30  # repeated for demonstration

    print("Training an 'expert' z-vector on a math-like dataset using RL.")
    rl_trainer.train_on_dataset(training_data_math, epochs=3)

    # Let's pretend we store this as our "math" expert
    z_vector_math = {}
    for pname, z_param in svf_wrapper.z_params.items():
        z_vector_math[pname] = z_param.detach().clone()

    # If we wanted multiple experts, we'd re-init or create separate wrappers, etc.
    # For brevity, let's imagine we have only "math" vs "others".
    # We'll define a random "others" z-vector to illustrate. 
    # In practice you would train it on a different dataset.
    z_vector_others = {}
    for pname, svd_comp in svd_dict.items():
        # random or identity
        z = torch.ones_like(svd_comp.S) * 0.05
        z_vector_others[pname] = z

    # 6) Self-adaptation library of experts
    z_vectors_library = {
        "math": z_vector_math,
        "others": z_vector_others
    }

    ########################################################
    # Example Inference: Prompt-based adaptation
    ########################################################
    # We'll do a 2-pass approach with a question
    sample_question = "Find the sum of 123 and 456."
    # First pass: ask classification
    adaptation_prompt = prompt_based_adaptation_llm(sample_question)
    classification_response = rl_trainer.generate(adaptation_prompt)
    # parse out \boxed{category}
    # This is a naive parse
    pred_class = "others"
    import re
    match = re.search(r"\\boxed\{(.*?)\}", classification_response)
    if match:
        pred_class = match.group(1).lower().strip()

    # Now we pick the z vector accordingly
    chosen_z_vector = adapt_select_z_vector(pred_class, z_vectors_library)
    if chosen_z_vector is None:
        chosen_z_vector = z_vectors_library["others"]

    # second pass: reconstruct the weight and generate the final
    # For demonstration, let's do a minimal approach:
    # We can patch the SVF wrapper z_params with chosen_z_vector
    with torch.no_grad():
        for pname, z_val in chosen_z_vector.items():
            svf_wrapper.z_params[pname].copy_(z_val)

    final_answer = rl_trainer.generate(sample_question, max_new_tokens=30)
    print("\n=== Prompt-based Adaptation ===")
    print(f"Classification response: {classification_response}")
    print(f"Predicted category: {pred_class}")
    print(f"Final answer: {final_answer}")

    ########################################################
    # Example Inference: Classifier-based adaptation
    ########################################################
    # In principle, you would train another small 'job classifier' expert
    # or a separate head. For brevity, we reuse prompt-based classification.

    ########################################################
    # Example Inference: Few-shot adaptation
    ########################################################
    # Let's say we have 5 test samples that are somewhat "math" style 
    # but not exactly the same domain.
    few_shot_data = [
        ("Compute 15+15?", "30"),
        ("What is 11 plus 9?", "20"),
        ("Compute 25 minus 4.", "21"),
        ("7 plus 10 = ?", "17"),
        ("What is 3+6?", "9"),
    ]

    # We'll do CEM search to find alpha for [math, others]
    # For demonstration, we skip the actual code that merges all param sets in real-time. 
    # Instead, we show the conceptual approach.

    # We'll build a minimal demonstration of alpha search for each param. 
    # In reality, you'd reconstruct each param from sum_{k} alpha_k * z^k

    # We'll just do a minimal approach: alpha for the entire set
    # i.e. W'(z) = W'(alpha_math * z_math + alpha_others * z_others)
    alpha_candidates = np.linspace(0, 1, 5)
    best_score = -999
    best_alpha = (0.5, 0.5)
    for alpha_m in alpha_candidates:
        alpha_o = 1 - alpha_m
        # Evaluate
        score = 0
        # Quick patch the model (naive)
        for pname in svf_wrapper.z_params.keys():
            new_z = z_vectors_library["math"][pname] * alpha_m + \
                    z_vectors_library["others"][pname] * alpha_o
            svf_wrapper.z_params[pname].data.copy_(new_z)

        # Evaluate correctness
        correct_count = 0
        for (q, a) in few_shot_data:
            gen_ans = rl_trainer.generate(q)
            if a in gen_ans:
                correct_count += 1
        curr_score = correct_count / len(few_shot_data)
        if curr_score > best_score:
            best_score = curr_score
            best_alpha = (alpha_m, alpha_o)
    print("\n=== Few-shot Adaptation (grid search example) ===")
    print(f"Best alpha found: math={best_alpha[0]:.2f}, others={best_alpha[1]:.2f} with score={best_score:.2f}")

if __name__ == "__main__":
    main()
