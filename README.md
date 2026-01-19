# CLI-Tuner

**Security-aware ML pipeline for translating natural language to Bash commands**

> Data and training pipeline for Qwen2.5-Coder-7B-Instruct with zero-tolerance validation

---

## What This Project Demonstrates

**Purpose:** Production-quality ML engineering workflow for secure LLM fine-tuning,
created as a Ready Tensor LLM Engineering certification project.

**What You'll Learn:**
- Security-first data preprocessing with zero-tolerance dangerous pattern filtering
- QLoRA fine-tuning on consumer GPUs (Qwen2.5-Coder-7B-Instruct)
- Comprehensive safety evaluation methodology
- Reproducible ML workflows with full provenance tracking

**Real-World Context:**
DevOps engineers frequently ask LLMs to generate Bash commands but face a trust
problem: models may hallucinate dangerous operations (`rm -rf /`, `chmod 777 /`)
without warning. This project demonstrates the ML engineering principles needed
to approach this problem safely.

**Key Results:**

| Metric | Result | Notes |
| --- | --- | --- |
| Training data safety | 0 dangerous commands in 1,735 filtered examples | Training-time filtering |
| Exact match accuracy | 13.22% (base 0%) | Strict string metric |
| Command-only rate | 99.4% (base ~30%) | Format learning |
| Adversarial safe rate | 57% (12/21) | Indicates need for runtime guardrails |

**Limitations:**
- Small training set (10% sample)
- Adversarial gaps show training-time filtering alone is insufficient

**Conclusion:**
This project demonstrates a complete, reproducible ML workflow with security-focused
design. Results show that production deployment would require additional safeguards
beyond training-time data filtering.

**Target Audience:** Early career AI/ML engineers, AI security practitioners
**Educational Materials:** Includes comprehensive lesson series (see `docs/lessons/`)

## Project Status

### Phase 0: Repository Setup (Complete)
- Directory structure, schemas, CI/CD, test scaffolding
- Zero-tolerance dangerous command patterns defined (17 patterns)
- Pydantic validation schemas for data pipeline

### Phase 1: Data Pipeline (Complete)
**Status:** Validated on 10% sample (1,835 examples, seed=42)

**Capabilities:**
- Field normalization (`nl_command` -> `instruction`, `bash_code` -> `output`)
- Shellcheck syntax validation (97.71% pass rate)
- Zero-tolerance dangerous pattern filtering (0 dangerous commands in output)
- Qwen2.5 chat template application (`<|im_start|>system/user/assistant<|im_end|>`)
- Assistant-only masking (user tokens masked to -100)
- Deduplication (58 duplicates removed)
- Provenance tracking (full audit trail with SHA256 hash)

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

### Phase 2: Training (Complete)
- QLoRA fine-tuning (rank=8, alpha=16, target modules: q_proj/v_proj/k_proj/o_proj)
- Final training loss: 1.0949 (W&B run sud23155)
- Final eval loss: 0.8840 (W&B run sud23155)
- Model checkpoint: `models/checkpoints/phase2-final/`
- W&B: https://wandb.ai/mwill-itmission20/cli-tuner/runs/sud23155
- Status: Loss charts attached (PDFs in docs/images: phase2_training_loss.pdf, phase2_eval_loss.pdf; optional GPUmetrics.png)
- Results doc: `docs/phase2_training_results.md`

### Phase 2: Reproducibility Artifacts

**Dataset Provenance:**
- Training data: `data/processed/train.jsonl` (1,388 examples)
- Dataset hash: `sha256:fba6d7048ededa4b728bd374d971ec3c62f4e93281280c175851a082b9d8a9bb`
- Full provenance: `data/processed/provenance.json`
- Fingerprint file: `docs/run_metadata/dataset_fingerprint.txt`

**Training Environment:**
- Platform: RunPod cloud GPU (24GB VRAM; exact model not recorded)
- CUDA: 12.1
- PyTorch: 2.5.1
- Transformers: 4.57.6
- Axolotl: 0.6.0
- PEFT: 0.14.0
- Full snapshot: `docs/run_metadata/environment_snapshot.txt`

**W&B Tracking:**
- Run ID: sud23155
- Run URL: https://wandb.ai/mwill-itmission20/cli-tuner/runs/sud23155
- Metadata: `docs/run_metadata/wandb_run_metadata.txt`

**Checkpoints:**
- Location: `models/checkpoints/phase2-final/`
- Training curves: `docs/images/phase2_training_loss.pdf`
- GPU metrics: `docs/images/GPUmetrics.png`

### Phase 3: Evaluation (Partial Pass)
- Report: `docs/phase3_evaluation_results.md`
- Domain: exact match 13.22% (base 0%), command-only 99.43%
- Generation success: 174/174 (100%)
- Safety: test set 0 dangerous commands (PASS); adversarial 12/21 safe (57% - FAIL)
- Status: inference-time guardrails required before deployment

### Phase 3: Evaluation Context

**Model Performance:**

- **Exact Match Accuracy:** 13.22% (base model: 0%)
- **Command-Only Format:** 99.4% (base model: ~30%)
- **Dangerous Commands Generated:** 0/174 (PASS)
- **Adversarial Resistance:** 12/21 safe (57% - FAIL)

**What This Demonstrates:**
This project demonstrates a complete ML engineering workflow: data preprocessing, QLoRA fine-tuning, and rigorous evaluation. The 13.22% exact match accuracy reflects the strictness of string-based evaluation; functionally equivalent commands with different syntax (for example, `ls -la` vs `ls -al`) are counted as failures. The 99.4% command-only rate shows successful format learning. The model learned the task structure but requires inference-time guardrails before deployment.

**Known Limitations:**

| Limitation | Planned Mitigation |
| ---------- | ------------------ |
| 13.22% exact match accuracy | Acceptable for demonstration; production use requires inference-time guardrails |
| Adversarial prompts bypass (9/21) | Inference-time filtering planned post-training |
| No runtime guardrails | Deployment with input/output validation (Phase 5+) |
| Small training set (1,735 examples) | Full dataset training possible with more compute |

**Training Environment:**

- **Platform:** RunPod cloud GPU (24GB VRAM; exact model not recorded)
- **Framework:** Axolotl 0.6.0 + PEFT 0.14.0
- **Base Model:** Qwen2.5-Coder-7B-Instruct
- **Quantization:** 4-bit NF4 (QLoRA)

---

## What You Can Run

- Phase 1 preprocessing: `python scripts/preprocess_data.py` (requires shellcheck)
- Phase 3 evaluation: `python scripts/evaluate_domain.py` and `python scripts/evaluate_safety.py`
- Training configs are included; full training requires Linux + CUDA (Axolotl/Triton) and a GPU

## Quick Start

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

**Phase 2 (Training) Requirements:**
- **Linux Required:** Axolotl training framework requires Linux (Triton dependency)
- **Windows Users:** Use WSL2 for development/testing (see [Phase 2 Spec](docs/Phase_2_Training_SPEC.md))
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

Note: The Phase 3 report in this repo was generated from evaluation outputs. Re-running the scripts above will populate `evaluation/` with fresh outputs.

---

## Security Philosophy

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
- Zero dangerous commands in training data (verified on 10% sample)
- Every command Shellcheck-validated for syntax correctness
- Full provenance trail (SHA256 hash, filtering stats, versions)
- Reproducible sampling (seed=42)

This isn't theoretical. **Real data loss happens from generated commands. We prevent dangerous commands from entering training data.**

---

## Data Flow (Phase 1)

```
HuggingFace Dataset (18,357 examples)
  v
[1] Load + Field Mapping (nl_command->instruction, bash_code->output)
  v [Random Sample: 1,835 examples, seed=42]
  v
[2] Shellcheck Syntax Validation (1,793 passed, 42 failed)
  v
[3] Dangerous Pattern Filtering (0 dangerous commands found)
  v
[4] Qwen Chat Template (<|im_start|>system/user/assistant<|im_end|>)
  v
[5] Tokenize + Assistant-Only Masking (user tokens = -100)
  v
[6] Deduplicate (58 duplicates removed) + Split (80/10/10, seed=42)
  v
[7] Save Outputs + Provenance + Validation Samples
  v
data/processed/
|-- train.jsonl (1,388 examples)
|-- val.jsonl (173 examples)
|-- test.jsonl (174 examples)
`-- provenance.json (full audit trail)
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
```

---

## Documentation

### Specifications
- [docs/SPECIFICATION_INDEX.md](docs/SPECIFICATION_INDEX.md) - Master index, issue tracking, phase status
- [docs/CLI-Tuner_Northstar_FINAL.md](docs/CLI-Tuner_Northstar_FINAL.md) - Architectural vision (v4.0)
- [docs/Phase_0_Setup_SPEC.md](docs/Phase_0_Setup_SPEC.md) - Repository initialization
- [docs/Phase_1_Data_Pipeline_SPEC.md](docs/Phase_1_Data_Pipeline_SPEC.md) - Data preprocessing (complete)
- [docs/Phase_2_Training_SPEC.md](docs/Phase_2_Training_SPEC.md) - Training implementation (complete)
- [docs/phase2_training_results.md](docs/phase2_training_results.md) - Phase 2 training results
- [docs/phase3_evaluation_results.md](docs/phase3_evaluation_results.md) - Phase 3 evaluation report

### Lessons
- [docs/lessons/Lesson_01_Phase1_Data_Pipeline.md](docs/lessons/Lesson_01_Phase1_Data_Pipeline.md) - Phase 1 walkthrough
- [docs/lessons/Lesson_02_Phase2_Training.md](docs/lessons/Lesson_02_Phase2_Training.md) - Phase 2 walkthrough
- [docs/lessons/Lesson_03_Phase3_Evaluation.md](docs/lessons/Lesson_03_Phase3_Evaluation.md) - Phase 3 walkthrough

### Reviews
- docs/reviews/Overseer_Review_v4.0_2026-01-15.md`](docs/reviews/Overseer_Review_v4.0_2026-01-15.md) - Initial Northstar review

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

**Future work:** See "Future Enhancements" for planned guardrails and optional performance improvements.

---

## Future Enhancements

### Inference-Time Safety Guardrails
**Status:** Scoped for future development

The Phase 3 evaluation revealed that 9/21 adversarial prompts bypassed training-time
safety filters (57% safe rate). Production deployment would require runtime validation:

**Proposed Architecture:**
```
User Input -> [Input Validator] -> [Model] -> [Output Validator] -> Safe Output
                    |                             |
            (reject malicious)           (catch dangerous)
```

**Key Features:**
- Input sanitization: Detect prompt injection attempts
- Output filtering: Re-apply 17 dangerous patterns to generated commands
- Confidence scoring: Flag low-confidence generations for review
- Audit logging: Track all generations with risk scores

**Target Success Criteria:**
- Adversarial safe rate: >=95% (currently 57%)
- Zero dangerous commands in output (maintain current test set PASS)
- Latency overhead: <50ms per request

**Implementation Notes:**
Runtime guardrails would be implemented as middleware wrapping the fine-tuned
model inference, allowing reuse across deployment scenarios (API, CLI, batch).

**Timeline:** Post-RT certification

---

### Model Performance Improvements
(Optional learning enhancements)

**Training Data Scaling:**
- Current: 1,388 examples (10% sample)
- Potential: Scale to full dataset (18,357 raw examples; filtered count TBD)
- Expected: 20-30% accuracy improvement

**Evaluation Metrics:**
- Current: Exact string match
- Potential: Semantic equivalence (AST comparison)

**Model Architecture:**
- Current: LoRA rank 8
- Potential: Rank 16 or full fine-tuning

---

## Educational Material

This project includes production-quality lessons for early career AI/ML engineers:

- **[Lesson 1: Security-First Data Pipelines](docs/lessons/Lesson_01_Phase1_Data_Pipeline.md)**
  Learn to build data pipelines with zero-tolerance dangerous pattern filtering, shellcheck integration, and provenance tracking.

- **[Lesson 2: QLoRA Fine-Tuning on Consumer GPUs](docs/lessons/Lesson_02_Phase2_Training.md)**
  Master parameter-efficient fine-tuning with Axolotl, 4-bit quantization, and experiment tracking with Weights & Biases.

- **[Lesson 3: Rigorous Safety Evaluation](docs/lessons/Lesson_03_Phase3_Evaluation.md)**
  Build comprehensive evaluation suites covering domain accuracy, safety validation, and adversarial robustness testing.

**Target Audience:** Early career AI/AI security engineers
**Pedagogy:** Concept -> Code -> Hands-on -> Interview Prep

See [`docs/lessons/`](docs/lessons/) for full curriculum.


---

## Questions & Support

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

## License

MIT License - See LICENSE file for details

---

## Project Milestones

- **2026-01-15:** Phase 1 Data Pipeline validated by Overseer (10% sample, 1,735 total)
- **2026-01-17:** Phase 2 Training complete (QLoRA, loss 1.0949 -> 0.8840, W&B run sud23155)
- **2026-01-18:** Phase 3 evaluation report generated (partial pass, adversarial fail)
- **Next:** Implement inference-time guardrails and re-run safety eval
