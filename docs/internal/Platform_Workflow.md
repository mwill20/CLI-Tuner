# Platform_Workflow.md
## SecureCLI-Tuner — End-to-End Platform Workflow (Option B → Option A)

This document is the **single operational runbook** for building, training, evaluating, and publishing SecureCLI-Tuner using multiple platforms.

It is designed to be:
- checklist-driven
- repeatable
- evolvable
- professional-grade

This workflow intentionally separates **where you code**, **where you run**, **how you track**, and **how you publish**.

---

## Mental Model (Read Once)

- **VS Code** → Source of truth (code + configs + docs)
- **RunPod** → Disposable GPU execution environment
- **Weights & Biases (W&B)** → Experiment tracking + lineage
- **Hugging Face Hub (HF)** → Final model registry

You do *not* build everything everywhere.

---

## 1. VS Code (Local) — Authoring & Control Plane

Purpose: write, edit, version, and validate everything *before* GPU time.

### Checklist

- [ ] Clone SecureCLI-Tuner repository locally
- [ ] Open repository in VS Code
- [ ] Verify directory structure:
  - `configs/` (Axolotl YAML)
  - `eval/` (regression suite)
  - `config/eval_config.yaml`
  - `docs/`
- [ ] Review **Unit_5_Northstar_Upgrade.md**
- [ ] Confirm training config YAML exists and is correct
- [ ] Confirm eval config YAML exists and is correct
- [ ] Commit all changes to git (clean working tree)
- [ ] Push branch to remote (GitHub)

**Rule:**  
No GPU time until VS Code repo is clean and committed.

---

## 2. RunPod (Option B) — GPU Execution Environment

Purpose: run training and evaluation on a rented GPU.

### Pod Creation Checklist

- [ ] Log in to RunPod
- [ ] Create new pod
- [ ] Select GPU (16–24 GB VRAM target)
- [ ] Choose base image (PyTorch / CUDA compatible)
- [ ] Start pod
- [ ] Open web terminal or Jupyter interface

---

### Environment Setup (RunPod)

- [ ] Clone repository:
  ```bash
  git clone <your-repo-url>
  cd securecli-tuner
  ```
- [ ] Create Python virtual environment
- [ ] Install dependencies:
  - PyTorch (GPU)
  - Axolotl
  - transformers
  - peft
  - datasets
  - lm-eval-harness
  - wandb
  - huggingface_hub
- [ ] Export environment variables (if needed)
- [ ] Verify GPU availability:
  ```bash
  nvidia-smi
  ```

---

## 3. Weights & Biases (W&B) — Tracking & Lineage

Purpose: track training runs and evaluation outputs while iterating.

### One-Time Setup (RunPod)

- [ ] Authenticate:
  ```bash
  wandb login
  ```
- [ ] Confirm login succeeds

---

### During Training

- [ ] Enable W&B logging in Axolotl YAML
- [ ] Start training:
  ```bash
  axolotl train configs/<training_config>.yaml
  ```
- [ ] Confirm run appears in W&B dashboard
- [ ] Monitor loss curves and metadata

---

## 4. Evaluation (RunPod) — Regression Suite

Purpose: decide whether a model is ship-worthy.

### Checklist

- [ ] Locate trained adapter output directory
- [ ] Run regression suite:
  ```bash
  python eval/run_eval_suite.py --config config/eval_config.yaml
  ```
- [ ] Confirm outputs:
  - `results/metrics.json`
  - `results/report.md`
- [ ] Review ship/no-ship gates

---

## 5. Hugging Face Hub — Publishing & Distribution

Purpose: publish the chosen model version.

### Authentication

- [ ] Login:
  ```bash
  huggingface-cli login
  ```

---

### Adapter Publication Checklist

- [ ] Prepare adapter directory
- [ ] Ensure required files exist
- [ ] Create HF repo (if needed)
- [ ] Upload:
  ```bash
  huggingface-cli upload <path> --repo-id <user/repo> --repo-type model
  ```
- [ ] Verify 5-line load works
- [ ] Tag version in model card

---

## 6. Overlap & Shared Responsibilities

| Action | VS Code | RunPod | W&B | HF |
|------|--------|--------|-----|----|
| Code edits | ✅ | ❌ | ❌ | ❌ |
| Training | ❌ | ✅ | 🔶 | ❌ |
| Evaluation | ❌ | ✅ | 🔶 | ❌ |
| Tracking | ❌ | 🔶 | ✅ | ❌ |
| Publishing | ❌ | 🔶 | ❌ | ✅ |

---

## 7. Failure Drills

- Training OOM → reduce batch / seq len
- Eval fails → verify paths + config
- Publish fails → re-auth HF, check repo perms

---

## 8. Graduation Path — Option A (Later)

- SSH into RunPod
- Attach VS Code Remote
- Same workflow, better ergonomics

---

## Final Rule

- VS Code = think
- RunPod = execute
- W&B = remember
- HF = ship

This document evolves with the project.
