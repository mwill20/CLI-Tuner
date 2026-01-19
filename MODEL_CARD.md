# Model Card: CLI-Tuner (Qwen2.5-Coder-7B-Instruct QLoRA Adapter)

## Model Details

### Model Description

CLI-Tuner is a QLoRA fine-tuned adapter for translating natural language instructions to Bash commands. Built on Qwen2.5-Coder-7B-Instruct, it demonstrates a security-first approach to command generation with zero-tolerance dangerous pattern filtering in training data.

- **Developed by:** Ready Tensor LLM Engineering Course Project
- **Model type:** QLoRA adapter (LoRA rank 8, alpha 16)
- **Language:** English (natural language) to Bash (commands)
- **License:** MIT (adapter); see base model license for Qwen2.5-Coder-7B-Instruct
- **Finetuned from:** Qwen/Qwen2.5-Coder-7B-Instruct

### Model Sources

- **Repository:** This repository
- **Base Model:** [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- **Training Framework:** Axolotl 0.6.0 + PEFT 0.14.0
- **Training Logs:** [W&B Run sud23155](https://wandb.ai/mwill-itmission20/cli-tuner/runs/sud23155)

## Uses

### Direct Use

This adapter is intended for **educational and demonstration purposes** to show a complete ML engineering workflow:

1. Security-first data preprocessing
2. QLoRA fine-tuning on consumer GPUs
3. Rigorous evaluation with safety metrics

### Downstream Use

**NOT recommended for production use without additional safeguards.** The model achieves 13.22% exact match accuracy and requires inference-time guardrails (CommandRisk) before deployment.

### Out-of-Scope Use

- **Production command execution:** Do not execute generated commands without human review
- **Security-critical environments:** Model can be manipulated by adversarial prompts
- **Autonomous systems:** Requires human-in-the-loop verification

## Bias, Risks, and Limitations

### Known Limitations

| Limitation | Impact | Mitigation |
| ---------- | ------ | ---------- |
| 13.22% exact match accuracy | Most generated commands differ from ground truth | Use as suggestion, not authoritative output |
| Adversarial bypass (9/21 prompts) | Crafted prompts can elicit dangerous commands | Requires CommandRisk inference guardrails |
| Small training set (1,735 examples) | Limited command diversity | Full dataset (18K) training possible |
| No runtime guardrails | Generated commands not validated at inference | Phase 5 adds input/output filtering |

### Recommendations

Users should:

1. **Never execute generated commands without review**
2. **Implement inference-time guardrails** before any deployment
3. **Treat outputs as suggestions**, not verified commands
4. **Test thoroughly** with adversarial prompts before use

## Training Details

### Training Data

- **Source:** `prabhanshubhowal/natural_language_to_linux` (HuggingFace)
- **Original Size:** 18,357 examples
- **Sampled Size:** 1,835 examples (10%, seed=42)
- **After Shellcheck:** 1,793 examples (42 invalid removed)
- **After Deduplication:** 1,735 examples (58 duplicates removed)
- **Final Split:** 1,388 train / 173 val / 174 test

### Preprocessing

1. Field normalization (`nl_command` -> `instruction`, `bash_code` -> `output`)
2. Shellcheck syntax validation (97.71% pass rate, 42 removed)
3. Zero-tolerance dangerous pattern filtering (17 patterns, 0 found)
4. Deduplication (58 duplicates removed)
5. Qwen2.5 chat template application
6. Assistant-only masking (user tokens = -100)

### Training Procedure

#### Training Hyperparameters

- **Training regime:** QLoRA (4-bit NF4 quantization)
- **LoRA Configuration:**
  - Rank: 8
  - Alpha: 16
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05
- **Optimizer:** AdamW (8-bit)
- **Learning rate:** 2e-4
- **Effective batch size:** 4 (micro batch size 1, gradient accumulation 4)
- **Epochs:** 3
- **Warmup:** 10 steps

#### Training Environment

- **Platform:** RunPod cloud GPU (24GB VRAM)
- **Hardware:** Exact GPU model not recorded (RunPod typically provisions A5000/A6000/A100)
- **Software:**
  - PyTorch 2.5.1
  - Transformers 4.57.6
  - PEFT 0.14.0
  - Axolotl 0.6.0
  - CUDA 12.1

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **Domain evaluation:** 174 held-out test examples
- **Adversarial evaluation:** 21 crafted prompts targeting 17 dangerous patterns

#### Metrics

| Metric | Description |
| ------ | ----------- |
| Exact Match Accuracy | Percentage of outputs matching ground truth exactly |
| Command-Only Rate | Percentage of outputs containing only a command (no explanation) |
| Dangerous Commands | Count of outputs matching zero-tolerance patterns |
| Adversarial Safe Rate | Percentage of adversarial prompts that do NOT produce dangerous output |

### Results

#### Domain Evaluation

| Metric | Base Model | Fine-tuned | Improvement |
| ------ | ---------- | ---------- | ----------- |
| Exact Match Accuracy | 0.0% | 13.22% | +13.22% |
| Command-Only Rate | ~30% | 99.43% | +69.43% |
| Successful Generations | 174/174 | 174/174 | - |

#### Safety Evaluation

| Test Set | Result | Status |
| -------- | ------ | ------ |
| Test data (174 examples) | 0 dangerous commands | PASS |
| Adversarial prompts (21) | 12/21 safe (57%) | FAIL |

**Overall Status:** PARTIAL PASS - Requires inference-time guardrails before deployment.

### Model Examination

The fine-tuned model successfully learned:

1. **Format compliance:** 99.4% command-only outputs (vs ~30% base)
2. **Task understanding:** Generates syntactically plausible Bash commands
3. **Training data safety:** Zero dangerous patterns in test set outputs

The model failed to learn:

1. **Adversarial resistance:** 9/21 adversarial prompts bypass safety
2. **Exact command matching:** Only 13.22% exact match (expected for precise command generation)

## Technical Specifications

### Model Architecture

- **Base:** Qwen2.5-Coder-7B-Instruct (7B parameters)
- **Adapter:** LoRA layers on attention projections
- **Adapter Parameters:** ~4M trainable (0.06% of base)

### Compute Infrastructure

#### Hardware

- RunPod cloud GPU (24GB VRAM; exact model not recorded)

#### Software

- Python 3.10+
- PyTorch 2.5.1
- Transformers 4.57.6
- PEFT 0.14.0
- Axolotl 0.6.0

## Citation

```bibtex
@misc{cli-tuner-2026,
  title={CLI-Tuner: Security-First LLM Pipeline for Bash Command Generation},
  author={Ready Tensor LLM Engineering Course},
  year={2026},
  howpublished={GitHub Repository},
}
```

## Model Card Authors

Ready Tensor LLM Engineering Course Project

## Model Card Contact

See repository issues for questions and feedback.
