# CLI-Tuner

**Secure LLM pipeline for translating natural language to Bash commands with security guardrails**

> Data and training pipeline for Qwen2.5-Coder-7B-Instruct with zero-tolerance validation

---

## 🎯 The Problem We're Solving

### Pain Point: Unsafe Command Generation for DevOps Teams

**The Customer Problem:**
- DevOps engineers spend hours writing complex Bash commands from memory or Stack Overflow
- When they ask an LLM for help, it generates commands—but they can't trust them without manual review
- **The Risk:** Hallucinates catastrophic operations like `rm -rf /`, `chmod 777 /`, `dd if=/dev/zero of=/dev/sda`
- **The Cost:** Data loss, system compromise, compliance violations
- **The Workflow Pain:** Generate → Review → Execute → Pray

**What We're Building:**
CLI-Tuner is a security-first pipeline that prepares safe training data and produces a QLoRA fine-tuned adapter. Runtime guardrails (input/output validation) are planned for later phases.

---

## 📋 Project Status

### ✅ Phase 0: Repository Setup (Complete)
- Directory structure, schemas, CI/CD, test scaffolding
- Zero-tolerance dangerous command patterns defined (17 patterns)
- Pydantic validation schemas for data pipeline

### ✅ Phase 1: Data Pipeline (Complete)
**Status:** Validated on 10% sample (1,835 examples, seed=42)

**Capabilities:**
- ✅ Field normalization (`nl_command` → `instruction`, `bash_code` → `output`)
- ✅ Shellcheck syntax validation (97.71% pass rate)
- ✅ Zero-tolerance dangerous pattern filtering (0 dangerous commands in output)
- ✅ Qwen2.5 chat template application (`<|im_start|>system/user/assistant<|im_end|>`)
- ✅ Assistant-only masking (user tokens masked to -100)
- ✅ Deduplication (58 duplicates removed)
- ✅ Provenance tracking (full audit trail with SHA256 hash)

**Latest Run (seed=42):**
- Total downloaded: 18,357 examples
- Sampled: 1,835 (10%)
- Shellcheck: 1,793/1,835 passed (97.71%)
- Invalid syntax removed: 42
- Dangerous commands removed: 0
- Duplicates removed: 58
- **Final split:** 1,388 train / 173 val / 174 test (1,735 total)

**Sampling Notes:**
- To mix the 10% sample, change `SAMPLE_SEED` in `scripts/preprocess_data.py` and rerun.
- Keep `SAMPLE_SIZE = 1835` for quick validation runs.

**Outputs:**
- `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/processed/provenance.json` (full audit trail)
- `data/validation/random_sample_50.jsonl` (dangerous pattern verification)
- `data/validation/chat_template_sample_10.txt` (manual review)

**Known Issues:**
- None. The previous `datetime.utcnow()` deprecation warning is resolved.

### ? Phase 2: Training (Complete)
- QLoRA fine-tuning (rank=8, alpha=16, target modules: q_proj/v_proj/k_proj/o_proj)
- Final training loss: 1.0949 (W&B run sud23155)
- Final eval loss: 0.8840 (W&B run sud23155)
- Model checkpoint: `models/checkpoints/phase2-final/`
- W&B: https://wandb.ai/mwill-itmission20/cli-tuner/runs/sud23155
- Status: Loss charts attached (PDFs in docs/images: phase2_training_loss.pdf, phase2_eval_loss.pdf; optional GPUmetrics.png)
- Results doc: `docs/phase2_training_results.md`

### Phase 3: Evaluation (Partial Pass)
- Report: `docs/phase3_evaluation_results.md`
- Domain: exact match 13.22% (base 0%), command-only 99.43%
- Generation success: 174/174 (100%)
- Safety: test set 0 dangerous commands (PASS); adversarial 12/21 safe (57% - FAIL)
- Status: inference-time guardrails required before deployment (CommandRisk in Phase 5)

### Phase 4-7: Deployment, Documentation, Submission (Planned)
- Inference guardrails (input/output validation)
- Evaluation report finalization + model card
- FastAPI deployment with guardrails
- Ready Tensor submission artifacts

### Phase 3: Evaluation Context

**Model Performance:**

- **Exact Match Accuracy:** 13.22% (base model: 0%)
- **Command-Only Format:** 99.4% (base model: ~30%)
- **Dangerous Commands Generated:** 0/174 (PASS)
- **Adversarial Resistance:** 12/21 safe (57% - FAIL)

**What This Demonstrates:**
This project demonstrates a complete ML engineering workflow: data preprocessing, QLoRA fine-tuning, and rigorous evaluation. The 13.22% exact match accuracy reflects the challenge of precise command generation, while the 99.4% command-only rate shows successful format learning. The model learned the task structure but requires inference-time guardrails before deployment.

**Known Limitations:**

| Limitation | Planned Mitigation |
| ---------- | ------------------ |
| 13.22% exact match accuracy | Acceptable for demonstration; production use requires CommandRisk guardrails |
| Adversarial prompts bypass (9/21) | CommandRisk V2 with inference-time filtering (Phase 5) |
| No runtime guardrails | FastAPI deployment with input/output validation (Phase 5+) |
| Small training set (1,735 examples) | Full dataset training possible with more compute |

**Training Environment:**

- **Platform:** RunPod cloud GPU (A100 40GB)
- **Framework:** Axolotl 0.6.0 + PEFT 0.14.0
- **Base Model:** Qwen2.5-Coder-7B-Instruct
- **Quantization:** 4-bit NF4 (QLoRA)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# Install Shellcheck
# Ubuntu/Debian
sudo apt-get install shellcheck

# macOS
brew install shellcheck

# Windows (Chocolatey)
choco install shellcheck
```

**⚠️ Phase 2 (Training) Requirements:**
- **Linux Required:** Axolotl training framework requires Linux (Triton dependency)
- **Windows Users:** Use WSL2 for development/testing (see [Phase 2 Spec](docs/Phase_2_Training_SPEC.md#️-platform-requirements))
- **Recommended:** RunPod cloud GPU for actual training (no local GPU needed)

### Run Phase 1 Pipeline

```bash
# Clone and setup
cd /path/to/cli-tuner
pip install -r requirements.txt

# Run preprocessing (10% sample)
python scripts/preprocess_data.py

# Generate validation samples
python scripts/generate_validation_sample.py
```

**Expected Output:**
```
Step 1/7: Loading dataset...
  Sampling 1835 examples from 18357 total (seed=42)
Step 2/7: Running shellcheck...
  Validating 1835 commands with shellcheck...
  Shellcheck complete: 1793/1835 passed (97%)
Step 3/7: Filtering dangerous commands...
Step 4/7: Applying chat template...
Step 5/7: Tokenizing with assistant-only masking...
Step 6/7: Splitting train/val/test...
  Removed 58 duplicate examples
Step 7/7: Saving outputs...
Pipeline complete. 1388 train, 173 val, 174 test.
```

### Configuration

Edit `scripts/preprocess_data.py`:
```python
SAMPLE_SIZE = 1835  # Set to None for full 18K dataset
SAMPLE_SEED = 42    # For reproducible sampling
SPLIT_SEED = 42     # For reproducible train/val/test split
```

### Debug Logging

Run scripts with `--debug` to write full debug logs to `logs/` with UTC timestamped filenames:

```
python scripts/preprocess_data.py --debug
python scripts/generate_validation_sample.py --debug
```

### Run Phase 3 Evaluation (after training)

Note: `scripts/verify_eval_dependencies.sh` is a bash script. Run it from WSL or Git Bash on Windows.

```
bash scripts/verify_eval_dependencies.sh

python scripts/evaluate_domain.py \
  --checkpoint models/checkpoints/phase2-final \
  --test-data data/processed/test.jsonl \
  --output-dir evaluation/domain

python scripts/evaluate_safety.py \
  --checkpoint models/checkpoints/phase2-final \
  --test-data data/processed/test.jsonl \
  --adversarial-prompts data/adversarial/adversarial_prompts.jsonl \
  --output-dir evaluation/safety

python scripts/compare_models.py \
  --checkpoint models/checkpoints/phase2-final \
  --test-data data/processed/test.jsonl \
  --output-dir evaluation/comparison

python scripts/generate_eval_report.py \
  --evaluation-dir evaluation \
  --output evaluation/reports/phase3_evaluation_results.md
```

Note: The Phase 3 report in this repo was generated from archived metrics under `docs/phase3_evaluation_results/metrics.json/metrics.json/`. Re-running the scripts above will populate `evaluation/` with fresh outputs.

---

## 🔐 Security Philosophy

**Zero-Trust Architecture:**
We treat every component as untrusted until validated:

1. **User input** -> runtime guardrails planned (Phase 5)
2. **Model output** -> evaluation implemented (Phase 3); runtime guardrails planned (Phase 5)
3. **Training data** -> validated for malicious examples (Phase 1 complete)
4. **Deployment** -> validation planned (Phase 5+)

**Zero-Tolerance Dangerous Patterns (17 patterns):**
```python
- Root deletion: rm -rf /, rm --recursive --force /
- Fork bombs: :(){:|:&};:
- Disk wipes: dd if=/dev/zero of=/dev/sd*
- Permission bombs: chmod -R 777 /
- Blind remote execution: curl | bash, wget | sh
- Direct disk writes: > /dev/sd*
- Filesystem formatting: mkfs.*, mkswap
```

**Phase 1 Guarantees:**
- ✅ Zero dangerous commands in training data (verified on 10% sample)
- ✅ Every command Shellcheck-validated for syntax correctness
- ✅ Full provenance trail (SHA256 hash, filtering stats, versions)
- ✅ Reproducible sampling (seed=42)

This isn't theoretical. **Real data loss happens from generated commands. We prevent dangerous commands from entering training data.**

---

## 📊 Data Flow (Phase 1)

```
HuggingFace Dataset (18,357 examples)
  ↓
[1] Load + Field Mapping (nl_command→instruction, bash_code→output)
  ↓ [Random Sample: 1,835 examples, seed=42]
  ↓
[2] Shellcheck Syntax Validation (1,793 passed, 42 failed)
  ↓
[3] Dangerous Pattern Filtering (0 dangerous commands found)
  ↓
[4] Qwen Chat Template (<|im_start|>system/user/assistant<|im_end|>)
  ↓
[5] Tokenize + Assistant-Only Masking (user tokens = -100)
  ↓
[6] Deduplicate (58 duplicates removed) + Split (80/10/10, seed=42)
  ↓
[7] Save Outputs + Provenance + Validation Samples
  ↓
data/processed/
├── train.jsonl (1,388 examples)
├── val.jsonl (173 examples)
├── test.jsonl (174 examples)
└── provenance.json (full audit trail)
```

## Data Flow (Phase 3 Evaluation)

```
Inputs
  - data/processed/test.jsonl
  - data/adversarial/adversarial_prompts.jsonl
  - models/checkpoints/phase2-final/
  - Base model (Qwen2.5-Coder-7B-Instruct)

[Expected outputs when scripts are run]
[1] evaluate_domain.py -> evaluation/domain/results.jsonl + metrics.json
[2] evaluate_safety.py -> evaluation/safety/metrics.json
[3] compare_models.py -> evaluation/comparison/base_vs_finetuned.json
[4] generate_eval_report.py -> evaluation/provenance.json + reports/phase3_evaluation_results.md

Optional
  - evaluate_general.py -> evaluation/general/{finetuned,baseline}/results.json

[Archived report inputs in this repo]
  - docs/phase3_evaluation_results/metrics.json/metrics.json/metrics.json
  - docs/phase3_evaluation_results/metrics.json/metrics.json/base_vs_finetuned.json
  - docs/phase3_evaluation_results/metrics.json/metrics.json/adversarial_results.jsonl
  - docs/phase3_evaluation_results/metrics.json/metrics.json/results.jsonl
```

---

## 📚 Documentation

### Specifications
- [`docs/SPECIFICATION_INDEX.md`](docs/SPECIFICATION_INDEX.md) - Master index, issue tracking, phase status
- [`docs/CLI-Tuner_Northstar_FINAL.md`](docs/CLI-Tuner_Northstar_FINAL.md) - Architectural vision (v4.0)
- [`docs/Phase_0_Setup_SPEC.md`](docs/Phase_0_Setup_SPEC.md) - Repository initialization
- [`docs/Phase_1_Data_Pipeline_SPEC.md`](docs/Phase_1_Data_Pipeline_SPEC.md) - Data preprocessing (complete)
- [`docs/Phase_2_Training_SPEC.md`](docs/Phase_2_Training_SPEC.md) - Training implementation (complete)
- [`docs/phase2_training_results.md`](docs/phase2_training_results.md) - Phase 2 training results
- [`docs/phase3_evaluation_results.md`](docs/phase3_evaluation_results.md) - Phase 3 evaluation report

### Lessons
- [`docs/lessons/Lesson_01_Phase1_Data_Pipeline.md`](docs/lessons/Lesson_01_Phase1_Data_Pipeline.md) - Phase 1 walkthrough
- [`docs/lessons/Lesson_02_Phase2_Training.md`](docs/lessons/Lesson_02_Phase2_Training.md) - Phase 2 walkthrough
- [`docs/lessons/Lesson_03_Phase3_Evaluation.md`](docs/lessons/Lesson_03_Phase3_Evaluation.md) - Phase 3 walkthrough

### Reviews
- [`docs/reviews/Overseer_Review_v4.0_2026-01-15.md`](docs/reviews/Overseer_Review_v4.0_2026-01-15.md) - Initial Northstar review

---

## Project Structure

```
cli-tuner/
|-- configs/
|   |-- evaluation_config.yaml
|   |-- axolotl_config.yaml
|   |-- axolotl_smoke_test.yaml
|-- data/
|   |-- raw/                    # Optional dataset cache
|   |-- processed/              # Phase 1 outputs (train/val/test.jsonl)
|   |-- logs/                   # Filtering logs (shellcheck, dangerous, violations)
|   |-- validation/             # Overseer validation samples
|   |-- adversarial/            # Adversarial prompts for Phase 3
|-- evaluation/                 # Phase 3 outputs
|   |-- domain/
|   |-- safety/
|   |-- comparison/
|   |-- reports/
|-- logs/                       # Debug logs (scripts --debug)
|-- models/
|   |-- checkpoints/            # Training checkpoints (Phase 2)
|-- schemas/
|   |-- dataset.py              # BashCommandExample (Pydantic)
|   |-- request.py              # API request schemas
|   |-- response.py             # API response schemas
|   |-- evaluation.py           # Phase 3 evaluation schema
|-- guardrails/
|   |-- patterns.py             # Zero-tolerance dangerous patterns (17 patterns)
|-- scripts/
|   |-- preprocess_data.py      # Phase 1 pipeline
|   |-- generate_validation_sample.py  # Overseer validation artifacts
|   |-- validate_checkpoint.py  # Phase 2 validation
|   |-- evaluate_domain.py      # Phase 3 domain metrics
|   |-- evaluate_safety.py      # Phase 3 safety checks
|   |-- compare_models.py       # Phase 3 comparison
|   |-- generate_eval_report.py # Phase 3 report generation
|   |-- eval_utils.py           # Shared evaluation helpers
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- fixtures/
|-- docs/
|   |-- SPECIFICATION_INDEX.md  # Master index
|   |-- Phase_*_SPEC.md         # Phase specifications
|   |-- lessons/                # Educational content
|   |-- phase3_evaluation_results/  # Archived Phase 3 metrics for the report
|-- requirements.txt            # Core dependencies
|-- requirements-eval.txt       # Phase 3 evaluation dependencies
```

---

## Learning Outcomes

**By completing Phase 1, you've mastered:**
- Field normalization and schema validation (Pydantic)
- Syntax validation with external tools (Shellcheck subprocess integration)
- Security-first data filtering (zero-tolerance patterns)
- LLM chat template application (Qwen2.5 format)
- Tokenization strategies (assistant-only masking)
- Data leakage prevention (deduplication before splitting)
- Provenance tracking (audit trails, SHA256 hashing)
- Reproducible sampling (fixed seeds for determinism)

**By completing Phase 2, you've learned:**
- QLoRA fine-tuning on consumer GPUs
- Axolotl configuration and training execution
- W&B experiment tracking and checkpoint management

**By completing Phase 3, you've validated:**
- Domain accuracy improvements over base model
- Command-only format compliance
- Safety gaps under adversarial prompts

**Phase 4+ will add:**

- Inference-time guardrails (CommandRisk) and safety re-evaluation
- Optional general benchmarks (GSM8K, HumanEval)
- Deployment with runtime guardrails (FastAPI)
- Documentation and Ready Tensor submission materials

---

## Future Enhancements: CommandRisk V2

The Phase 3 evaluation revealed that 9/21 adversarial prompts bypassed safety filters. **CommandRisk V2** addresses this gap with inference-time guardrails.

### Planned Architecture

```text
User Input
    |
    v
[Input Validator] -- reject malicious prompts
    |
    v
[Fine-tuned Model] -- generate command
    |
    v
[Output Validator] -- catch dangerous patterns
    |
    v
Safe Command Output (or rejection message)
```

### Key Features

| Feature | Description | Status |
| ------- | ----------- | ------ |
| Input sanitization | Detect prompt injection attempts | Planned (Phase 5) |
| Output filtering | Apply 17 dangerous patterns to generated output | Planned (Phase 5) |
| Confidence scoring | Flag low-confidence generations for review | Planned (Phase 5) |
| Audit logging | Track all generations with risk scores | Planned (Phase 5) |

### Success Criteria

- Adversarial safe rate: >= 95% (currently 57%)
- Zero dangerous commands in output (maintain current PASS)
- Latency overhead: < 50ms per request

### Implementation Notes

CommandRisk V2 will be implemented as a FastAPI middleware layer that wraps the fine-tuned model inference. This allows the same guardrails to be reused across different deployment scenarios (API, CLI, batch processing).

---

## 🎓 Educational Material

This project includes production-quality educational content:
- **Target audience:** Early career AI/AI security engineers
- **Pedagogy:** Concept → Code → Hands-on → Interview Prep
- **Current lessons:** Phase 1 Data Pipeline, Phase 2 Training, Phase 3 Evaluation (see docs/lessons/)

See [`docs/lessons/`](docs/lessons/) for full curriculum.

---

## ✉️ Questions & Support

**Built with:**
- Ready Tensor LLM Engineering & Deployment certification course
- Production security engineering best practices
- Real customer problems as the north star

**Design philosophy:**
- Every design choice is documented
- Every tradeoff is explained
- Every component has a purpose

**Start with the pain points. Build the solution. Ship with confidence.**

---

## Attribution

- Dataset: `prabhanshubhowal/natural_language_to_linux` (HuggingFace). See the dataset card for license and terms.
- Base model: `Qwen2.5-Coder-7B-Instruct`. See the model card for license and terms.
- Tooling: `shellcheck` for Bash syntax validation.
- Program: Ready Tensor LLM Engineering & Deployment certification course.

---

## 📝 License

MIT License - See LICENSE file for details

---

## ?? Project Milestones

- ? **2026-01-15:** Phase 1 Data Pipeline validated by Overseer (10% sample, 1,735 total)
- ? **2026-01-17:** Phase 2 Training complete (QLoRA, loss 1.0949 -> 0.8840, W&B run sud23155)
- ? **2026-01-18:** Phase 3 evaluation report generated (partial pass, adversarial fail)
- ?? **Next:** Implement CommandRisk guardrails (Phase 5) and re-run safety eval
