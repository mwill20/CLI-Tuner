# Axolotl Training & Reproducibility Guide
## SecureCLI‑Tuner Engineering Standard

## Purpose
This document defines **Axolotl as the default training system** for SecureCLI‑Tuner and future LLM fine‑tuning projects.

Axolotl enables:
- reproducible experiments
- configuration‑driven workflows
- clean separation of code and training logic
- production‑aligned practices

---

## Why Axolotl

Without Axolotl:
- Training logic spreads across scripts
- Configurations drift
- Experiments become hard to reproduce

With Axolotl:
- One YAML defines everything
- Switching strategies requires editing config keys
- Experiments are version‑controllable artifacts

This mirrors how professional LLM teams operate.

---

## What Axolotl Controls

A single YAML file defines:

- Base model
- Dataset & format
- Tokenization & sequence length
- Precision & quantization
- LoRA / QLoRA configuration
- Optimizer & scheduler
- Batch sizes & accumulation
- Distributed strategy (DDP / ZeRO / FSDP)
- Logging & checkpoints

The YAML is the **source of truth**.

---

## SecureCLI‑Tuner Standard Usage

### Required
- QLoRA (`load_in_4bit: true`)
- LoRA adapters
- Mixed precision
- Gradient checkpointing
- Sample packing

### Optional (Documented)
- DDP for speed
- ZeRO‑2 or FSDP for future scaling

---

## Example: SecureCLI‑Tuner QLoRA Skeleton

```yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
load_in_4bit: true
adapter: lora

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

datasets:
  - path: prabhanshubhowal/natural_language_to_linux
    type: alpaca

sequence_len: 2048
sample_packing: true

micro_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-4
optimizer: adamw_8bit

bf16: auto
gradient_checkpointing: true
```

---

## Reproducibility Rules

- Every experiment = one YAML
- YAMLs are versioned in Git
- No ad‑hoc script edits
- Results must be reproducible from config alone

---

## Long‑Term Standard

Axolotl is not “just for this project.”

It is the **default fine‑tuning interface** going forward:
- SecureCLI‑Tuner
- Future CLI agents
- Security‑focused LLMs
- Guardrail‑augmented models

This standard is **locked**.

---

## Final Note

Axolotl doesn’t replace:
- Transformers
- PEFT
- DeepSpeed
- FSDP

It **organizes them**.

That distinction is the difference between experimentation and engineering.
