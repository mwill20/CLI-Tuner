
# Qwen2.5-Coder-7B — GPU Memory Profile for Training and Fine-Tuning
This document applies Week 4’s LLM Engineering memory framework directly to **Qwen2.5-Coder-7B**, the model used in the SecureCLI-Tuner project.

It demonstrates **why full fine-tuning is impossible on consumer GPUs** and why **QLO RA is the correct engineering choice** for our project.

---

# 1. Assumptions
- Model: **Qwen2.5-Coder-7B**
- Parameter count (N): **~7,000,000,000**
- Optimizer: **Adam** (2 FP32 states)
- Sequence length for activations estimate: **512**
- Batch size: small (to match realistic 16–24GB VRAM GPUs)
- Activation checkpointing assumed
- 1 GB ≈ (10^9) bytes for back-of-envelope calculations

---

# 2. Memory Components Refresher
Training memory ≈  
**Parameters + Gradients + Optimizer States + Activations + Overhead**

Where:

| Component | Required During Training? | Notes |
|----------|----------------------------|-------|
| Parameters | ✔ | Base model weights |
| Gradients | ✔ | Same dtype as parameters |
| Optimizer States | ✔ | Adam stores 2 FP32 tensors per param |
| Activations | ✔ | Grow with batch size + sequence length |
| Framework Overhead | ✔ | CUDA buffers, memory fragmentation |

---

# 3. Scenario A — Full Fine-Tuning, FP32

### 3.1 Parameters (FP32)
7B params × 4 bytes  
= **28 GB**

### 3.2 Gradients (FP32)
Same size as weights  
= **28 GB**

### 3.3 Optimizer States (Adam – FP32)
Each parameter stores:
- m (4 bytes)
- v (4 bytes)

7B × 8 bytes  
= **56 GB**

### 3.4 Activations
For seq=512, small batch, checkpointing:  
≈ **8 GB**

### 3.5 Framework / CUDA Overhead
≈ **4 GB**

### ✔ FP32 Total

**28 + 28 + 56 + 8 + 4 = ~124 GB VRAM**

---

# 4. Scenario B — Full Fine-Tuning, FP16/BF16

### 4.1 Parameters (FP16)
7B × 2 bytes  
= **14 GB**

### 4.2 Gradients (FP16)
Same size  
= **14 GB**

### 4.3 Optimizer States (Adam – FP32)
7B × 8 bytes  
= **56 GB**

### 4.4 Activations
FP16 reduces activation size slightly  
≈ **6 GB**

### 4.5 Overhead
≈ **4 GB**

### ✔ FP16/BF16 Total

**14 + 14 + 56 + 6 + 4 = ~94 GB VRAM**

---

# 5. Scenario C — QLoRA Fine-Tuning (What SecureCLI-Tuner Uses)

QLoRA changes the entire memory equation.

### 5.1 Quantized Base Weights (4-bit)
4-bit = 0.5 bytes/param  
7B × 0.5 bytes  
≈ **3.5 GB** (rounded to **~4 GB** with overhead)

### 5.2 Trainable LoRA Parameters
Typical LoRA trains **~0.3%** of parameters.

7B × 0.003  
= **21M trainable parameters**

Breakdown:

| Component | Memory |
|----------|--------|
| LoRA weights (FP16) | 21M × 2 bytes = **0.042 GB** |
| LoRA gradients (FP16) | **0.042 GB** |
| LoRA optimizer states (FP32) | 21M × 8 bytes = **0.168 GB** |

**Total LoRA overhead ≈ 0.25 GB**

### 5.3 Activations
Activations still dominate training:  
≈ **6 GB**

### 5.4 Overhead
≈ **2 GB**

### ✔ QLoRA Total

**4 + 0.25 + 6 + 2 = ~12.25 GB VRAM**

---

# 6. Summary Table

| Training Method | GPU VRAM Required | Feasible on 24GB GPU? |
|----------------|-------------------|------------------------|
| Full FT (FP32) | ~124 GB | ❌ Impossible |
| Full FT (FP16/BF16) | ~94 GB | ❌ Impossible |
| QLoRA (4-bit + adapters) | ~12–16 GB | ✔ Yes |

---

# 7. Why This Matters for SecureCLI-Tuner
Our project must be reproducible, practical, and aligned with real hardware constraints.

Thus:

- Full fine-tuning is *not* feasible.  
- QLoRA is the **correct engineering strategy** for Qwen2.5-Coder-7B.  
- This choice is grounded in quantifiable GPU memory math—not preference.  

This section will be included in:
- `MODEL_CARD.md`  
- Ready Tensor publication  
- Repository `docs/` directory  

---

# 8. Recommended Citation in Documentation

> “A full FP32 fine-tune of Qwen2.5-Coder-7B would require ~124GB VRAM.  
> A mixed-precision fine-tune still requires ~94GB.  
> Using QLoRA, we reduce the training footprint to ~12–16GB, enabling  
> single-GPU fine-tuning on commodity 16–24GB GPUs.  
> This memory analysis directly informed the architectural design of SecureCLI-Tuner.”

---

# 9. End of Document  
*This analysis will be updated if training configuration, sequence length, or LoRA rank changes.*
