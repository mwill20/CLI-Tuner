---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
- axolotl
- base_model:adapter:Qwen/Qwen2.5-Coder-7B-Instruct
- lora
- transformers
datasets:
- data/processed/train.jsonl
pipeline_tag: text-generation
model-index:
- name: models/checkpoints
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.12.2`
```yaml
# CLI-Tuner Phase 2 Training Configuration
# QLoRA fine-tuning of Qwen2.5-Coder-7B-Instruct
# Target: 24GB consumer GPU (RTX 3090/4090)

# ============================================================================
# BASE MODEL
# ============================================================================
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: false

# ============================================================================
# QUANTIZATION (4-bit NF4 for memory efficiency)
# ============================================================================
load_in_4bit: true
adapter: lora

# ============================================================================
# LORA CONFIGURATION
# ============================================================================
lora_model_dir: null
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
lora_target_linear: false
lora_fan_in_fan_out: false

# ============================================================================
# DATASET
# ============================================================================
datasets:
  - path: data/processed/train.jsonl
    type: completion
    field: text

# Validation dataset (using separate file from Phase 1)
val_set_size: 0.0  # Don't split from train - use test_datasets instead

test_datasets:
  - path: data/processed/val.jsonl
    split: train
    type: completion
    field: text

# ============================================================================
# SEQUENCE CONFIGURATION
# ============================================================================
sequence_len: 512
sample_packing: false
pad_to_sequence_len: true

# ============================================================================
# BATCH SIZE & GRADIENT ACCUMULATION
# ============================================================================
micro_batch_size: 1
gradient_accumulation_steps: 4
eval_batch_size: 1

# Effective batch size: micro_batch_size × gradient_accumulation_steps = 4

# ============================================================================
# TRAINING DURATION
# ============================================================================
num_epochs: 1
max_steps: 500  # ~1.4 epochs for 1,388 train examples

# ============================================================================
# OPTIMIZER & LEARNING RATE
# ============================================================================
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002
warmup_steps: 50  # 10% of max_steps

# ============================================================================
# PRECISION & MEMORY OPTIMIZATION
# ============================================================================
bf16: true
fp16: false
tf32: false
gradient_checkpointing: true

# ============================================================================
# CHECKPOINTING
# ============================================================================
output_dir: models/checkpoints
save_strategy: steps
save_steps: 100
save_total_limit: 3  # Keep last 3 checkpoints only

# ============================================================================
# EVALUATION
# ============================================================================
eval_strategy: steps
eval_steps: 50

# ============================================================================
# LOGGING
# ============================================================================
logging_steps: 10

# ============================================================================
# WEIGHTS & BIASES
# ============================================================================
wandb_project: cli-tuner
wandb_run_name: phase2-training-v1
wandb_log_model: "checkpoint"

# ============================================================================
# SPECIAL TOKENS (Qwen format - already in tokenizer)
# ============================================================================
special_tokens: {}

```

</details><br>

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/mwill-itmission20/cli-tuner/runs/05kwtar3)
# models/checkpoints

This model is a fine-tuned version of [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) on the data/processed/train.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8815
- Memory/max Mem Active(gib): 8.5
- Memory/max Mem Allocated(gib): 8.5
- Memory/device Mem Reserved(gib): 10.77

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Use OptimizerNames.ADAMW_BNB with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 50
- training_steps: 500

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Mem Active(gib) | Mem Allocated(gib) | Mem Reserved(gib) |
|:-------------:|:------:|:----:|:---------------:|:---------------:|:------------------:|:-----------------:|
| No log        | 0      | 0    | 4.0502          | 8.5             | 8.5                | 10.77             |
| 1.5486        | 0.1441 | 50   | 1.4786          | 8.5             | 8.5                | 10.77             |
| 0.9021        | 0.2882 | 100  | 0.9451          | 8.5             | 8.5                | 10.77             |
| 0.8819        | 0.4323 | 150  | 0.9239          | 8.5             | 8.5                | 10.77             |
| 0.9907        | 0.5764 | 200  | 0.9083          | 8.5             | 8.5                | 10.77             |
| 0.8788        | 0.7205 | 250  | 0.8975          | 8.5             | 8.5                | 10.77             |
| 0.9251        | 0.8646 | 300  | 0.8899          | 8.5             | 8.5                | 10.77             |
| 0.8624        | 1.0086 | 350  | 0.8843          | 8.5             | 8.5                | 10.77             |
| 0.8423        | 1.1527 | 400  | 0.8838          | 8.5             | 8.5                | 10.77             |
| 0.8264        | 1.2968 | 450  | 0.8817          | 8.5             | 8.5                | 10.77             |
| 0.8305        | 1.4409 | 500  | 0.8815          | 8.5             | 8.5                | 10.77             |


### Framework versions

- PEFT 0.17.0
- Transformers 4.55.2
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.21.4