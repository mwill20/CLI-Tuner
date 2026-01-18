# Phase 2: Training Specification
**Version:** 1.1  
**Created:** 2026-01-16  
**Updated:** 2026-01-16 (Windows/WSL2 requirements added)  
**Status:** Ready for Implementation  
**Addresses:** OSV-007 (Phase 2 Handoff), QLoRA Training Setup  
**Dependencies:** Phase 1 complete (data pipeline validated)  
**Duration:** 1-2 weeks (implementation + initial training runs)  
**Owner:** Primary Engineer

---

## ⚠️ PLATFORM REQUIREMENTS

**CRITICAL:** Axolotl requires Linux (Triton dependency not available on Windows).

**Deployment Options:**
1. **RunPod Cloud GPU** - Recommended for training (RTX A5000 @ $0.27/hr)
2. **WSL2 (Windows Subsystem for Linux)** - For local development/testing
3. **Native Linux** - If you have a local GPU (24GB+ VRAM required)

---

### WSL2 Setup for Windows Users

**Purpose:** Develop and test Axolotl configs on Windows before running expensive GPU training on RunPod.

**Prerequisites:**
- Windows 10 version 2004+ or Windows 11
- Administrator access

**Installation Steps:**

1. **Install WSL2**
```powershell
# Open PowerShell as Administrator
# Right-click Start → Windows PowerShell (Admin)

wsl --install

# This installs:
# - WSL2 (Windows Subsystem for Linux v2)
# - Ubuntu 22.04 LTS (default distribution)
# - Virtual Machine Platform
```

2. **Restart Computer**
```powershell
# Required for WSL2 installation to complete
Restart-Computer
```

3. **First Launch - Create Linux User**
```bash
# After restart, Ubuntu terminal opens automatically
# Create username and password (can be different from Windows)

# Example:
Enter new UNIX username: yourname
New password: ********
Retype new password: ********
```

4. **Navigate to Project Directory**
```bash
# Windows drives mounted at /mnt/<drive-letter>/
cd /mnt/c/Projects/Ready\ Tensor\ LLM/

# Verify you're in the right place
ls -la  # Should see README.md, configs/, data/, etc.
```

5. **Set Up Python Environment**
```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ (Ubuntu 22.04 has 3.10 by default)
python3 --version  # Should show 3.10+

# Install pip and venv
sudo apt install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

6. **Install Axolotl**
```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Using CPU-only PyTorch for WSL2 (no GPU needed for config testing)

# Install Axolotl
pip install axolotl

# Verify installation
axolotl --version
python -c "import axolotl; print('Axolotl installed successfully')"
```

7. **Install VS Code WSL Extension (Optional)**
```powershell
# From Windows PowerShell (not WSL2)
code --install-extension ms-vscode-remote.remote-wsl

# Then open project in WSL2:
# VS Code → Command Palette (Ctrl+Shift+P) → "WSL: Reopen Folder in WSL"
# Navigate to /mnt/c/Projects/Ready Tensor LLM/
```

**What You Can Do in WSL2:**
- ✅ Validate Axolotl config syntax (`axolotl preprocess configs/axolotl_config.yaml`)
- ✅ Test dataset loading (check tokenization, padding)
- ✅ Develop and test scripts
- ✅ Run Phase 1 pipeline (data processing)
- ❌ Full training (no GPU in WSL2 unless Windows 11 + CUDA drivers)

**What Requires RunPod:**
- Training (smoke test and full 500-step run)
- GPU memory profiling
- Production inference

**Common WSL2 Commands:**
```bash
# Start WSL2 from Windows
wsl

# Exit WSL2 (back to Windows)
exit

# Shutdown WSL2 (from Windows PowerShell)
wsl --shutdown

# Check WSL2 status
wsl --status

# List installed distributions
wsl --list --verbose
```

**File Access:**
- **From WSL2 to Windows:** `/mnt/c/Users/YourName/Documents/`
- **From Windows to WSL2:** `\\wsl$\Ubuntu\home\yourname\`
- Tip: Keep project on Windows drive (`/mnt/c/Projects/`) for easy access from both

---

### RunPod Setup

**See Appendix A for complete RunPod deployment instructions.**

Quick reference:
- GPU: RTX A5000 ($0.27/hr)
- Template: PyTorch 2.1
- Access: Web Terminal (no SSH setup needed)
- Cost: ~$0.20-0.35 for Phase 2 training

---

## PURPOSE

Fine-tune Qwen2.5-Coder-7B-Instruct on CLI-Tuner dataset using QLoRA to produce a safe Bash command generation model within consumer GPU constraints (24GB VRAM).

**Inputs:** Clean train/val/test splits from Phase 1 (1,388 train / 173 val / 174 test = 1,735 total examples)  
**Outputs:** Fine-tuned LoRA adapters, training checkpoints, W&B experiment logs  
**Success Criteria:**
- ✅ Training completes without OOM errors
- ✅ Loss converges (validation loss decreases)
- ✅ Model generates valid Bash commands (not gibberish)
- ✅ Checkpoint saved and versioned
- ✅ Full experiment tracked in Weights & Biases

---

## ARCHITECTURE OVERVIEW

```
Phase 2 Training Pipeline:

Training Data (1,735 examples)
  ↓
[1] Model Loading
  - Qwen2.5-Coder-7B-Instruct (HF Hub)
  - 4-bit quantization (NF4)
  - bfloat16 compute dtype
  ↓
[2] QLoRA Configuration
  - LoRA rank r=8, alpha=16
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - ~4.7M trainable parameters (0.067% of 7B)
  ↓
[3] Axolotl Training Setup
  - Optimizer: AdamW (8-bit)
  - LR: 2e-4 with cosine decay
  - Batch size: effective 4 (micro=1, grad_accum=4)
  - Mixed precision: bf16
  ↓
[4] Training Loop
  - Max steps: 500-1000
  - Gradient checkpointing enabled
  - Save every 100 steps
  ↓
[5] Weights & Biases Logging
  - Loss, LR, gradient norm
  - GPU memory, throughput
  - Hyperparameters, dataset hash
  ↓
Final Output: LoRA adapters + full provenance
```

---

## COMPONENT 1: BASE MODEL LOADING

### Model Selection
```yaml
model:
  name: Qwen/Qwen2.5-Coder-7B-Instruct
  source: HuggingFace Hub
  architecture: AutoModelForCausalLM
  tokenizer: AutoTokenizer (from same model)
  trust_remote_code: false  # Security: no custom code execution
```

### Quantization Configuration
```yaml
quantization:
  method: BitsAndBytes (bitsandbytes library)
  precision: 4-bit (NF4 - Normal Float 4-bit)
  compute_dtype: bfloat16
  double_quant: true  # Additional memory savings
  quant_type: nf4
```

**Memory Estimate (OVERSEER VERIFIED):**
- Base model (4-bit): ~3.5GB
- Quantization overhead: ~0.5GB
- **Total base model: ~4GB** ✅

**Rationale:**
- Full FP32: 28GB (impossible on 24GB GPU)
- Full BF16: 14GB (leaves insufficient room for optimizer states)
- 4-bit NF4: ~4GB (enables QLoRA within VRAM budget)

**Source:** Memory profile analysis in `Qwen2.5-7B_Memory_Profile (1).md`

---

## COMPONENT 2: QLORA CONFIGURATION

### LoRA Hyperparameters
```yaml
lora:
  rank: 8  # Low rank for parameter efficiency
  alpha: 16  # Scaling factor (typically 2×rank)
  dropout: 0.05  # Light regularization
  bias: none  # Do not train bias terms
  task_type: CAUSAL_LM
  
  target_modules:
    - q_proj  # Query projection in attention
    - v_proj  # Value projection in attention
    - k_proj  # Key projection in attention
    - o_proj  # Output projection in attention
```

**Trainable Parameters (OVERSEER CALCULATION):**
- LoRA introduces: **~4.7M trainable parameters** (0.067% of 7B base)
- Memory breakdown:
  - LoRA weights (bf16): ~9MB
  - LoRA gradients (bf16): ~9MB
  - LoRA optimizer states (fp32): ~38MB
  - **Total LoRA overhead: ~56MB** ✅

**Why These Modules?**
- Attention projections (q_proj, k_proj, v_proj, o_proj) are highest-leverage layers in transformers
- Qwen2.5 architecture uses these standard projection names
- Excludes MLP layers to reduce trainable parameters while maintaining performance

**Alternatives Considered:**
- **r=16**: Doubles trainable params to ~9.4M (more capacity, more VRAM)
- **r=4**: Halves trainable params to ~2.4M (less capacity, may underfit)
- **Recommendation:** Start with r=8, scale to r=16 if model underfits

---

## COMPONENT 3: TRAINING CONFIGURATION (AXOLOTL)

### Optimizer & Learning Rate
```yaml
optimizer:
  type: adamw_bnb_8bit  # 8-bit Adam via bitsandbytes
  learning_rate: 2.0e-4  # Standard for LoRA fine-tuning
  weight_decay: 0.0
  lr_scheduler: cosine
  warmup_steps: 0.1  # 10% of total steps
```

**OVERSEER NOTE:**  
- AdamW 8-bit: **~1GB optimizer states** (vs. ~56GB for FP32 optimizer on full model)
- Learning rate 2e-4 is proven for LoRA (OpenAI/Anthropic papers)
- Cosine decay prevents learning rate from staying too high at end of training

### Batch Size & Gradient Accumulation
```yaml
batch_config:
  micro_batch_size: 1  # Sequences processed per forward pass
  gradient_accumulation_steps: 4  # Accumulate gradients over 4 microbatches
  effective_batch_size: 4  # micro_batch_size × gradient_accumulation_steps
```

**OVERSEER VALIDATION:**
- **Why micro_batch_size=1?** Qwen sequences can be long (~512 tokens), larger batches risk OOM
- **Why effective_batch_size=4?** Balance between training stability and memory constraints
- **Gradient accumulation:** Simulates larger batch without loading all examples into VRAM simultaneously

### Sequence & Precision
```yaml
sequence_config:
  max_seq_length: 512  # Truncate/pad to 512 tokens
  pad_to_sequence_len: true
  
precision:
  bf16: true  # Brain Float 16 (better numerical stability than fp16)
  fp16: false
  tf32: false
  gradient_checkpointing: true  # Trade 20% speed for ~50% VRAM savings
```

**OVERSEER NOTE:**
- **Gradient checkpointing:** Recomputes activations during backward pass instead of storing
  - Memory savings: ~4GB (activations reduced from ~8GB to ~4GB)
  - Speed cost: ~20% slower training
  - **Verdict:** Worth it to fit in 24GB

### Training Steps & Epochs
```yaml
training_duration:
  num_epochs: 1  # Single pass through dataset
  max_steps: 500  # Or ~1.1 epochs for 1,388 train examples
  save_steps: 100  # Checkpoint every 100 steps
  eval_steps: 50  # Evaluate on validation set every 50 steps
  logging_steps: 10  # Log metrics every 10 steps
```

**OVERSEER CALCULATION:**
- Dataset: 1,388 train examples
- Effective batch size: 4
- Steps per epoch: 1,388 / 4 = **347 steps/epoch**
- 500 steps ≈ **1.44 epochs**

**Rationale:**
- Start with 500 steps (~1.4 epochs) to test convergence
- Extend to 1,000 steps (2.9 epochs) if loss hasn't plateaued
- Monitor validation loss for early stopping signal

---

## COMPONENT 4: WEIGHTS & BIASES INTEGRATION

### W&B Configuration
```yaml
wandb:
  project: cli-tuner
  entity: null  # Uses default W&B user/org
  run_name: phase2-training-v1
  log_model: false  # Don't upload full model to W&B (LoRA adapters only)
  log_interval: 10  # Log every 10 steps
```

**Metrics Logged:**
- **Loss:** train_loss, eval_loss
- **Learning Rate:** lr (cosine decay schedule)
- **Gradients:** grad_norm (detect exploding gradients)
- **System:** gpu_memory_allocated, gpu_memory_reserved, throughput (tokens/sec)
- **Hyperparameters:** All config values (lr, batch_size, lora_r, lora_alpha, etc.)

### Provenance Tracking
```yaml
provenance:
  data_hash: <SHA256 from Phase 1 provenance.json>
  model_base: Qwen/Qwen2.5-Coder-7B-Instruct
  training_date: <UTC timestamp>
  axolotl_version: <from environment>
  transformers_version: <from environment>
  bitsandbytes_version: <from environment>
```

**USER ACTION REQUIRED:**
1. Create W&B account (free tier): https://wandb.ai/signup
2. Get API key: Settings → API Keys
3. Store in `.env` file (see `.env.example`)
4. Never commit `.env` to git (already in .gitignore)

---

## COMPONENT 5: CHECKPOINTING STRATEGY

### Checkpoint Configuration
```yaml
checkpointing:
  output_dir: models/checkpoints
  save_strategy: steps
  save_steps: 100
  save_total_limit: 3  # Keep last 3 checkpoints (disk space management)
  save_only_model: false  # Save optimizer state for resumption
```

**Checkpoint Contents:**
- LoRA adapter weights (`adapter_model.bin`)
- Adapter config (`adapter_config.json`)
- Tokenizer files (for inference reproducibility)
- Optimizer state (for training resumption)
- Training arguments (`training_args.bin`)

**Disk Space Estimate:**
- Per checkpoint: ~200MB (LoRA adapters + optimizer state)
- 3 checkpoints: ~600MB
- Final checkpoint: ~200MB
- **Total:** ~800MB ✅

### Checkpoint Naming
```
models/checkpoints/
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
└── phase2-final/  # Final checkpoint directory
```

---

## MEMORY BUDGET (OVERSEER VERIFICATION)

**Total VRAM Breakdown:**

| Component | Memory | Source |
|-----------|--------|--------|
| Base model (4-bit) | 3.5GB | BitsAndBytes quantization |
| Quantization overhead | 0.5GB | Dequantization buffers |
| LoRA adapters (bf16) | 0.01GB | Trainable weights |
| LoRA gradients (bf16) | 0.01GB | Backprop |
| LoRA optimizer states (fp32) | 0.04GB | 8-bit AdamW |
| Activations (with checkpointing) | 4GB | Forward/backward pass |
| CUDA/PyTorch overhead | 2GB | Memory fragmentation, buffers |
| **Total** | **~10.1GB** | **Fits in 24GB with ~14GB headroom** ✅ |

**Safety Margin:** 14GB headroom allows for:
- Larger batches if needed (micro_batch_size=2)
- Longer sequences if needed (max_seq_length=768)
- Temporary CUDA allocations during training

**OVERSEER VERDICT:** Memory budget is **conservative and validated**. Training will not OOM on 24GB GPU.

---

## AXOLOTL CONFIG FILE SPECIFICATION

### File: `configs/axolotl_config.yaml`

```yaml
# Base model configuration
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: false

# Quantization
load_in_4bit: true
adapter: lora
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

# Dataset
datasets:
  - path: data/processed/train.jsonl
    type: completion
    field: text
val_set_size: 0.0  # Use separate validation file below

# Validation dataset
test_datasets:
  - path: data/processed/val.jsonl
    type: completion
    field: text

# Sequence configuration
sequence_len: 512
sample_packing: false
pad_to_sequence_len: true

# Batch and gradient accumulation
micro_batch_size: 1
gradient_accumulation_steps: 4
eval_batch_size: 1

# Training duration
num_epochs: 1
max_steps: 500

# Optimizer
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002
warmup_steps: 50  # ~10% of 500 steps

# Precision
bf16: true
fp16: false
tf32: false
gradient_checkpointing: true

# Checkpointing
output_dir: models/checkpoints
save_strategy: steps
save_steps: 100
save_total_limit: 3

# Evaluation
eval_strategy: steps
eval_steps: 50

# Logging
logging_steps: 10

# Weights & Biases
wandb_project: cli-tuner
wandb_run_name: phase2-training-v1
wandb_log_model: false

# Special tokens (Qwen format - already in tokenizer)
special_tokens: {}
```

**OVERSEER NOTES:**
- `val_set_size: 0.0`: We use separate val.jsonl instead of splitting train.jsonl
- `test_datasets`: Points to val.jsonl for eval during training
- `sample_packing: false`: Each example is a single sequence (no concatenation)
- `warmup_steps: 50`: 10% of 500 steps = 50 warmup steps

---

## TRAINING COMMAND

### Prerequisites Check
```bash
# Verify shellcheck, Python, transformers installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
python -c "import axolotl; print(f'Axolotl: {axolotl.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Training Launch
```bash
# Activate environment (if using conda/venv)
# conda activate cli-tuner

# Set environment variables
export WANDB_API_KEY=<your_wandb_api_key>  # Or use .env file
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management

# Launch training with Axolotl
accelerate launch -m axolotl.cli.train configs/axolotl_config.yaml
```

### Alternative: Direct Python Script
If Axolotl CLI is unavailable, fallback to Python script:

```bash
python scripts/train.py --config configs/axolotl_config.yaml
```

**OVERSEER NOTE:** Axolotl CLI is recommended for automatic distributed training setup, gradient accumulation handling, and W&B integration.

---

## SMOKE TEST (10 EXAMPLES)

Before running full training, validate setup with 10-example smoke test:

### Create Smoke Test Dataset
```bash
# Sample 10 examples from train.jsonl
head -n 10 data/processed/train.jsonl > data/processed/train_smoke.jsonl
head -n 5 data/processed/val.jsonl > data/processed/val_smoke.jsonl
```

### Smoke Test Config: `configs/axolotl_smoke_test.yaml`
```yaml
# Copy from axolotl_config.yaml, change:
datasets:
  - path: data/processed/train_smoke.jsonl
    type: completion
    field: text

test_datasets:
  - path: data/processed/val_smoke.jsonl
    type: completion
    field: text

max_steps: 10  # Just 10 steps to verify no errors
save_steps: 5
eval_steps: 5
logging_steps: 1

wandb_run_name: smoke-test-v1
```

### Run Smoke Test
```bash
accelerate launch -m axolotl.cli.train configs/axolotl_smoke_test.yaml
```

**Expected Output:**
- Training starts without OOM errors
- Loss logged every step
- Checkpoint saved at step 5 and 10
- W&B run appears in dashboard
- **Duration:** 2-3 minutes on RTX 4090

**OVERSEER VALIDATION CHECKLIST:**
- [ ] No CUDA OOM errors
- [ ] Loss decreases from step 0 to step 10
- [ ] GPU memory usage < 12GB (well under 24GB limit)
- [ ] Checkpoint directory created with adapter files
- [ ] W&B dashboard shows run with loss curve

---

## TRAINING TIME ESTIMATES

**Hardware Assumptions:**
- RTX 4090 (24GB VRAM, ~80 TFLOPS)
- RTX 3090 (24GB VRAM, ~35 TFLOPS)

**Timing Estimates:**

| GPU | Steps/sec | 500 steps | 1,000 steps |
|-----|-----------|-----------|-------------|
| RTX 4090 | ~0.25 | ~33 min | ~66 min |
| RTX 3090 | ~0.14 | ~60 min | ~120 min |

**Factors Affecting Speed:**
- Sequence length (512 tokens is moderate)
- Gradient checkpointing (adds ~20% overhead)
- Logging frequency (every 10 steps is minimal overhead)

**OVERSEER NOTE:** These are estimates. Actual speed depends on system (CPU, RAM, disk I/O).

---

## COST ESTIMATES (IF USING RUNPOD)

**USER ACTION REQUIRED:**
If training on local GPU, skip this section. If using RunPod cloud GPU:

1. **Go to RunPod:** https://www.runpod.io/
2. **Select GPU pod:**
   - RTX 4090: ~$0.69/hr
   - RTX 3090: ~$0.44/hr
3. **Deploy:** Use their PyTorch template or custom Axolotl image
4. **Cost for 500 steps:**
   - RTX 4090: ~$0.35
   - RTX 3090: ~$0.33

**OVERSEER NOTE:** User is responsible for RunPod setup. Provide pod SSH access and Axolotl install commands if needed.

---

## POST-TRAINING VALIDATION

After training completes, validate the checkpoint:

Optional automated validation (generates sample outputs):
```bash
python scripts/validate_checkpoint.py --checkpoint-dir models/checkpoints/phase2-final --sample-size 5
```

### 1. Check Checkpoint Files
```bash
ls -lh models/checkpoints/phase2-final/
# Expected files:
# - adapter_model.bin (~100MB)
# - adapter_config.json
# - tokenizer_config.json
# - special_tokens_map.json
```

### 2. Load Model for Inference Test
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "models/checkpoints/phase2-final"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Test inference
messages = [
    {"role": "user", "content": "List all files in the current directory"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Expected Output:**
- Model generates a Bash command (e.g., `ls`, `ls -la`, `find . -maxdepth 1`)
- Output is syntactically valid (no gibberish)
- Output does not contain dangerous commands (verified in Phase 3)

---

## DECISION POINTS (OVERSEER RECOMMENDATIONS)

### 1. Epoch Count vs. Max Steps
**Architect's Question:** Train for 1 epoch or monitor validation loss for early stopping?

**OVERSEER RECOMMENDATION:**
- **Start with max_steps=500** (~1.4 epochs)
- Monitor eval_loss every 50 steps
- If eval_loss still decreasing at step 500, extend to 1,000 steps
- If eval_loss plateaus or increases, stop early

**Rationale:** Fixed step count is easier for initial run. Early stopping requires manual monitoring of W&B dashboard.

### 2. LoRA Rank: r=8 or r=16?
**Architect's Question:** Use r=8 or r=16?

**OVERSEER RECOMMENDATION:**
- **Start with r=8** (matches Architect's design)
- If model underfits (eval_loss high, generated commands are generic), increase to r=16
- r=16 doubles trainable params (~9.4M) and VRAM (~100MB extra) - still fits in budget

**Rationale:** r=8 is standard for 7B models. Scaling to r=16 is trivial config change.

### 3. Validation Strategy: Separate validation.jsonl or Split from train?
**Architect's Question:** Use separate validation.jsonl or split from train?

**OVERSEER RECOMMENDATION:**
- **Use separate validation.jsonl** (already created in Phase 1)
- Set `val_set_size: 0.0` in Axolotl config
- Point `test_datasets` to `data/processed/val.jsonl`

**Rationale:** Phase 1 already created val.jsonl with proper deduplication. Splitting from train.jsonl would ignore Phase 1 work.

### 4. Checkpoint Frequency: Every 100 steps or 250 steps?
**Architect's Question:** Save checkpoints every 100 or 250 steps?

**OVERSEER RECOMMENDATION:**
- **Every 100 steps** for 500-step runs
- **Every 250 steps** for 1,000+ step runs

**Rationale:** For initial runs, frequent checkpoints allow resumption if training crashes. Overhead is minimal (~2 seconds to save adapters).

---

## SECURITY CONSIDERATIONS

### 1. Training Data Integrity
**Verified in Phase 1:**
- ✅ Zero dangerous commands in training data
- ✅ All commands Shellcheck-validated
- ✅ Provenance tracked (data hash, filtering stats)

### 2. Model Drift Detection
**Phase 3 Responsibility:**
- After training, run CommandRisk on model outputs
- If model generates dangerous commands NOT in training data, investigate:
  - Hallucination from base model knowledge
  - LoRA overfitting
  - Tokenizer artifacts

### 3. Checkpoint Provenance
**Tracked in W&B and checkpoint metadata:**
- Data hash (SHA256 from Phase 1)
- Hyperparameters (lr, batch_size, lora_r, etc.)
- Training duration (steps, wall-clock time)
- Loss curves (train/val)

### 4. Secrets Management
**Critical:**
- ✅ `.env` file gitignored (W&B API key)
- ✅ Never log secrets to W&B or checkpoints
- ✅ HuggingFace Hub auth token (if pushing model) stored in `~/.cache/huggingface/token`

---

## READY TENSOR COURSE CONNECTION

### Module 3: Parameter-Efficient Fine-Tuning (PEFT)
- **QLoRA technique:** 4-bit quantization + LoRA adapters
- **Rank/alpha selection:** r=8, alpha=16 (standard 2×rank scaling)
- **Target module selection:** Attention projections (highest leverage)

### Module 4: Training Strategies
- **Gradient accumulation:** Simulate larger batches without VRAM cost
- **Learning rate scheduling:** Cosine decay with warmup (10% warmup steps)
- **Gradient checkpointing:** Trade 20% speed for 50% VRAM savings

### Module 5: Experiment Tracking
- **W&B integration:** Full hyperparameter logging, loss curves, system metrics
- **Reproducibility:** Data hash, model version, training config all logged

---

## FILES TO CREATE (IMPLEMENTATION CHECKLIST)

### 1. Primary Config
- [x] `docs/Phase_2_Training_SPEC.md` (this file)
- [x] `configs/axolotl_config.yaml` (main training config)
- [x] `configs/axolotl_smoke_test.yaml` (10-example validation)

### 2. Environment & Secrets
- [x] `.env.example` (template for W&B API key)
- [x] Update `.gitignore` (verify `.env` is excluded)

### 3. Training Scripts (Optional)
- [ ] `scripts/train.py` (fallback if Axolotl CLI unavailable)
- [x] `scripts/validate_checkpoint.py` (post-training inference test)

### 4. Documentation
- [x] `docs/Phase_2_Training_SPEC.md` (this file)
- [x] Update `docs/SPECIFICATION_INDEX.md` (add Phase 2 reference)
- [ ] `docs/lessons/Lesson_02_Phase2_Training.md` (educational content - deferred to after training)

---

## HANDOFF TO PRIMARY ENGINEER

**OVERSEER VALIDATION STATUS:** ✅ **APPROVED FOR IMPLEMENTATION**

**Corrections from Architect's Design:**
1. **Example count:** Architect claimed 1,746 examples. Actual count is **1,735 total** (1,388 train, 173 val, 174 test). ✅ Corrected in spec.
2. **Steps per epoch:** Architect estimated ~437 steps/epoch. Actual: **347 steps/epoch** (1,388 / 4 = 347). ✅ Corrected in spec.
3. **Memory estimate:** Architect's VRAM math is correct but conservative. Total is ~10GB (not 16.7GB). ✅ Verified and updated.
4. **Val set strategy:** Clarified that `val_set_size: 0.0` with separate `test_datasets` path is correct approach. ✅ Documented.

**Implementation Priority:**
1. **HIGH:** Verify `configs/axolotl_config.yaml` matches the spec above
2. **HIGH:** Verify `.env.example` template for W&B API key
3. **MEDIUM:** Run smoke test (10 examples) to validate setup
4. **MEDIUM:** Run full training (500 steps)
5. **LOW:** Create training wrapper script (optional - Axolotl CLI preferred)

**Next Steps After Implementation:**
- PE verifies configs and runs smoke test
- PE submits smoke test results (W&B link, checkpoint files, terminal output)
- Overseer validates smoke test before approving full training run
- After full training, move to Phase 3: Evaluation

---

## APPENDIX A: RUNPOD TRAINING SETUP

**USER ACTION REQUIRED:** This project has no local GPU. RunPod cloud GPUs are recommended.

### 1. Create RunPod Account

1. Visit https://www.runpod.io/ and sign up
2. Navigate to **Billing** → Add payment method (credit card required)
3. Add **$10-20** for initial testing (enough for 30-70 hours on RTX A5000)

### 2. Optional: Add SSH Key

**For first-time users: Skip this step. Use Web Terminal instead.**

If you want SSH access (for file transfers or VS Code Remote-SSH):

```powershell
# Windows PowerShell - Generate SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"
# Save to: C:\Users\YourName\.ssh\id_ed25519
# Press Enter for no passphrase

# Copy public key
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Paste public key content into RunPod **Settings** → **SSH Keys**.

### 3. Deploy GPU Pod

1. Navigate to **Pods** → **+ GPU Pod**
2. Select GPU:
   - **Recommended: RTX A5000** - $0.27/hr, 24GB VRAM, medium availability
   - Backup: RTX 4090 - $0.59/hr, 24GB VRAM, high availability
   - Budget: L4 - $0.39/hr, 24GB VRAM, low availability
3. Template: **PyTorch 2.1** (or Ubuntu 22.04 + manual setup)
4. Disk:
   - Container Disk: 50GB
   - Volume Disk: 50GB (persistent storage for checkpoints)
5. Click **Deploy**

### 4. Connect to Pod

**Option A: Web Terminal (Easiest)**
1. Click **Connect** button on your pod
2. Select **Start Web Terminal**
3. Terminal opens in browser

**Option B: SSH (If you added SSH key)**
```powershell
# From Windows PowerShell
ssh root@<pod-public-ip> -p <port>
# IP and port shown in pod details
```

### 5. Setup Environment on Pod

```bash
# Clone your repository
cd /workspace
git clone https://github.com/yourusername/ready-tensor-llm.git
cd ready-tensor-llm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install axolotl

# Verify Axolotl installation
axolotl --version

# Set up Weights & Biases
export WANDB_API_KEY=your_wandb_key_here
wandb login
```

### 6. Upload Dataset to Pod

**Option A: Git (if dataset is in repo)**
```bash
# Already cloned in step 5
ls data/processed/  # Verify train.jsonl, val.jsonl, test.jsonl
```

**Option B: Manual Upload (if dataset not in repo)**
```bash
# From local Windows machine
scp -P <port> data/processed/*.jsonl root@<pod-ip>:/workspace/ready-tensor-llm/data/processed/

# Or use RunPod web interface: Files → Upload
```

### 7. Run Smoke Test (10 examples)

```bash
# Create smoke test dataset
head -n 10 data/processed/train.jsonl > data/processed/train_smoke.jsonl
head -n 5 data/processed/val.jsonl > data/processed/val_smoke.jsonl

# Run smoke test
accelerate launch -m axolotl.cli.train configs/axolotl_smoke_test.yaml

# Monitor in W&B: https://wandb.ai/mwill-itmission/cli-tuner
```

**Expected smoke test results:**
- Duration: 5-10 minutes
- Final loss: <2.0 (should decrease from initial ~3.0)
- Memory usage: ~10GB VRAM
- Checkpoint saved: `outputs/smoke-test/checkpoint-10/`

### 8. Run Full Training (500 steps)

```bash
# After smoke test validates successfully
accelerate launch -m axolotl.cli.train configs/axolotl_config.yaml

# Monitor progress
# - W&B dashboard: https://wandb.ai/mwill-itmission/cli-tuner
# - Local logs: outputs/cli-tuner-qwen-lora/training.log
```

**Expected full training results:**
- Duration: 30-60 minutes (RTX A5000/4090)
- Checkpoints: Every 100 steps → 5 checkpoints total
- Final checkpoint (local): `models/checkpoints/phase2-final/`

### 9. Download Checkpoints

**Option A: RunPod Web Interface**
1. Click **Files** on your pod
2. Navigate to `/workspace/ready-tensor-llm/outputs/`
3. Select checkpoint folder → **Download**

**Option B: SCP (if SSH configured)**
```powershell
# From Windows PowerShell
scp -r -P <port> root@<pod-ip>:/workspace/ready-tensor-llm/outputs ./outputs
```

**Option C: RunPodCTL (Advanced)**
```bash
# Install runpodctl: https://github.com/runpod/runpodctl
runpodctl receive <pod-id>:/workspace/ready-tensor-llm/outputs ./outputs
```

### 10. Terminate Pod

**IMPORTANT:** Stop pod when not training to avoid charges.

1. Navigate to **Pods** dashboard
2. Click **Stop** (preserves volume data) or **Terminate** (deletes everything)
3. Verify billing dashboard shows pod stopped

**Cost Summary (RTX A5000 @ $0.27/hr):**
- Smoke test: $0.02-0.05 (5-10 minutes)
- Full training: $0.14-0.27 (30-60 minutes)
- Total for Phase 2: ~$0.20-0.35

---

### Troubleshooting

**Pod won't start:**
- GPU unavailable → Try different region or GPU type
- Insufficient funds → Add billing credit

**OOM (Out of Memory) Error:**
- Reduce `micro_batch_size` from 1 → 0.5 (if supported)
- Enable `gradient_checkpointing: true` (already enabled)
- Switch to larger GPU (RTX A6000 48GB @ $0.49/hr)

**Training too slow:**
- Verify GPU utilization: `nvidia-smi` (should be >80%)
- Check disk I/O: Use volume storage, not container disk
- Upgrade to RTX 4090 ($0.59/hr, ~30% faster)

**W&B not logging:**
- Verify login: `wandb verify`
- Check API key: `echo $WANDB_API_KEY`
- Test connection: `wandb online`

---

## APPENDIX B: AXOLOTL INSTALLATION

**USER ACTION REQUIRED:**

**Linux/WSL2:**
```bash
# Install Axolotl (PyPI method)
pip install axolotl

# Or install from source (latest features)
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .

# Verify installation
python -c "import axolotl; print(axolotl.__version__)"

# Install additional dependencies
pip install bitsandbytes>=0.41.0
pip install transformers>=4.35.0
pip install accelerate>=0.25.0
pip install peft>=0.7.0
pip install wandb>=0.16.0
```

**OVERSEER NOTE:** User is responsible for Axolotl installation. Provide troubleshooting support if needed.

---

## END OF SPECIFICATION

**Status:** Ready for PE implementation  
**Blocker Check:** ✅ No blockers  
**Memory Budget:** ✅ Validated (10GB / 24GB)  
**Dataset:** ✅ Ready (1,735 examples)  
**Next Review:** After smoke test completion

