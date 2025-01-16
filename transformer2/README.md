# Transformer²

This repository contains an end-to-end prototype for **Transformer²** (“Transformer-Squared”)—a self-adaptive Large Language Model (LLM) framework. It demonstrates:

- **Singular Value Fine-tuning (SVF)** for parameter-efficient specialization.  
- **PPO-based RL** for directly optimizing the latent \(\mathbf{z}\)-vectors.  
- **Multi-domain adaptivity** through different strategies:  
  1. **Prompt-based** classification,  
  2. **Classifier-based** (placeholder here), and  
  3. **Few-shot mixture** of experts via **CEM**.  

---

## Table of Contents

1. [Overview](#overview)  
2. [Features & Highlights](#features--highlights)  
3. [Repository Structure](#repository-structure)  
4. [Dependencies](#dependencies)  
5. [Usage](#usage)  
6. [Implementation Details](#implementation-details)  
   1. [SVD & SVFWrapper](#svd--svfwrapper)  
   2. [RL via PPO (with `trl`)](#rl-via-ppo-with-trl)  
   3. [Adaptation Strategies](#adaptation-strategies)  
7. [Extending & Customizing](#extending--customizing)  
8. [FAQ](#faq)  

---

## Overview

**Transformer²** is a research framework enabling LLMs to **dynamically adapt** to multiple domains or tasks by only modifying a minimal number of parameters—specifically the **singular values** of the model’s weight matrices. This approach is especially powerful when:

- You want to avoid full fine-tuning of billions of parameters.  
- You need to add or specialize for new tasks after pre-training.  
- You need the model to quickly adapt to changing requirements at inference time.  

In this production-oriented example, we demonstrate how to:

1. **Compute SVD** of select weight matrices from a base model.  
2. **Attach** small \(\mathbf{z}\)-vectors that scale those singular values to produce new weight matrices.  
3. **Use PPO** to optimize those \(\mathbf{z}\)-vectors for domain-specific tasks.  
4. **Perform adaptation** in real time for new tasks or domains.

---

## Features & Highlights

- **Parameter Efficiency**: We only learn small \(\mathbf{z}\)-vectors, leaving the vast majority of the base model frozen.  
- **Stable RL Fine-tuning**: Uses [**trl**](https://github.com/lvwerra/trl)'s PPO implementation, integrated with [**Accelerate**](https://github.com/huggingface/accelerate).  
- **Scalability**: The pipeline works for smaller or larger models. For very large models (e.g., 70B), you can rely on advanced parallelization.  
- **Modularity**: Adapt the code to new tasks or sub-domains by plugging in your own data, reward functions, and classification prompts.  

---

## Repository Structure

```
.
├── transformer2_production.py  # Main code illustrating Transformer² for production
├── README.md                   # This README file
└── requirements.txt            # Example dependencies (optional)
```

- **`transformer2_production.py`**  
  - **`SvdComponent`** & **SVD logic**  
  - **`SVFWrapper`** for storing and managing the \(\mathbf{z}\)-parameters  
  - **`SVF_PPOTrainer`** leveraging RL (PPO)  
  - **Adaptation** methods (prompt-based, few-shot w/ CEM, etc.)  
  - **CLI** setup with `argparse`  

---

## Dependencies

Below is a minimal set of packages to run this script:

- **Python** 3.10+  
- **PyTorch** 2.0+ (with CUDA if using GPU)  
- **Transformers** \(\geq 4.30.0\)  
- **Accelerate** \(\geq 0.18.0\)  
- **TRL** \(\geq 0.5.0\) (for PPO)  
- **numpy**, **scipy** (for sampling, truncated normal, stats)  

You can place them in a `requirements.txt`:

```
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.18.0
trl>=0.5.0
numpy
scipy
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Usage

1. **Clone** or copy this repository.  
2. **Install** the dependencies (above).  
3. **Run** `transformer2_production.py` from your terminal, for example:

   ```bash
   python transformer2_production.py \
     --model_name DeepSeekInstruct/large \
     --output_dir outputs \
     --logging_dir logs \
     --batch_size 4 \
     --max_epochs 1 \
     --learning_rate 1e-4
   ```

   Adjust flags as needed.

4. The script will:
   - Load the base model + tokenizer from `--model_name`  
   - Decompose the selected parameters with SVD.  
   - Wrap them in an `SVFWrapper`.  
   - Launch a PPO training loop on a small mocked “math domain” dataset.  
   - Save a checkpoint of the \(\mathbf{z}\)-vectors.  
   - Show how to do prompt-based adaptation on a sample question.  

5. Inspect results in the logs and the `outputs/` directory.

---

## Implementation Details

### 1. SVD & SVFWrapper

- **SVD** is computed for each parameter matrix \( W \) (e.g., MLP or attention weight).  
- We store `(U, S, V^T)` in a `SvdComponent`.  
- **`SVFWrapper`** holds:
  1. The base model in **frozen** mode.  
  2. A `nn.ParameterDict` of learned **\(\mathbf{z}\)** vectors, each of shape = rank(\(W\)).  
  3. A method `patch_weights()` that reconstructs \( W' = U \cdot \operatorname{diag}(S \cdot z) \cdot V^T \) to modify the base model’s parameters in-place.  

### 2. RL via PPO (with `trl`)

- We rely on **PPO** from [**trl**](https://github.com/lvwerra/trl) to train only the **\(\mathbf{z}\)-vectors**.  
- The script:
  1. Creates a deep copy of the base model as a “reference model” (frozen) for KL divergence.  
  2. Calls `PPOTrainer.step(...)` with `(query_tensors, response_tensors, rewards)` to update the policy.  
  3. Uses a naive substring-based reward in the example. In real usage, you can incorporate advanced checks, code execution, or specialized reward models.  

### 3. Adaptation Strategies

The code includes the building blocks for typical **Transformer²** adaptation:

1. **Prompt-based**:  
   - We generate a classification label for an incoming question using the same (or a specialized) LLM.  
   - We load or use the \(\mathbf{z}\)-vector that corresponds to the predicted domain.  

2. **Classifier-based**:  
   - Similar to prompt-based, but we might train a small classification head or LLM-based classifier.  

3. **Few-shot interpolation**:  
   - Uses a **Cross-Entropy Method** (CEM) or other optimization over possible alpha-coefficients that linearly combine multiple experts’ \(\mathbf{z}\)-vectors.  
   - This script includes a function `compute_cem_interpolation(...)`, which demonstrates how to do a global alpha search.  

---

## Extending & Customizing

1. **Larger Models**  
   - For 7B+ or 70B+ models, consider [FSDP](https://pytorch.org/docs/master/fsdp.html) or [DeepSpeed Zero](https://www.deepspeed.ai/) to handle memory scaling.  

2. **Custom Reward**  
   - For code tasks, you can run unit tests on the generated code and assign a reward for correct solutions.  
   - For knowledge tasks, you might use a specialized reward model.  

3. **Multiple Domain Experts**  
   - You can create multiple sets of \(\mathbf{z}\)-vectors (one per domain: math, code, reasoning, vision, etc.) and store them in a dictionary or separate `.pt` files.  
   - Combine them with the few-shot approach if a new domain arises that partially intersects with existing ones.  

4. **Logging & Monitoring**  
   - Switch PPO’s config: `log_with="wandb"` or `"tensorboard"` for real-time metric tracking.  
   - For advanced setups, log to custom DB or S3.  

5. **Deployment**  
   - After training, the \(\mathbf{z}\)-vectors are extremely small (megabytes or even kilobytes).  
   - Deploy them as sidecar “experts.” At inference time, do a quick 2-pass adaptation to pick or combine the correct domain expert.  

---

## FAQ

1. **Why do we only decompose certain layers?**  
   - Decomposing *every* layer can be expensive. Often, you decompose only MLP or attention projection layers. Empirically, even partial-layer SVF yields good gains and reduces overhead.

2. **How can I store \(\mathbf{z}\)-vectors for multiple tasks?**  
   - Either keep separate checkpoint files—e.g., `zparams_math.pt`, `zparams_code.pt`—or store them in a dictionary with keys for each domain.

3. **What if my model is too large for SVD on a single GPU?**  
   - Use [FAISS GPU-based SVD], or break the matrix into sub-blocks, or run CPU-based SVD with enough RAM, or do distributed SVD. Some people also do approximate SVD for extremely large weight matrices.

4. **Can I use LoRA or any other method in synergy?**  
   - Yes. In principle, you can combine LoRA with SVD-based parameterization. **Transformer²** is about dynamic adaptation; the exact parameterization can be flexible.

5. **Where do I place reward modeling?**  
   - Reward modeling can be integrated in the `SVF_PPOTrainer.train_on_prompts(...)` function. Instead of naive substring checks, you’d do more sophisticated scoring.
