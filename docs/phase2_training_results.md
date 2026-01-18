# Phase 2: Training Results

## W&B Tracking

**Run Link:** https://wandb.ai/mwill-itmission20/cli-tuner/runs/sud23155
**Project:** cli-tuner
**Run Name:** lilac-waterfall-5
**Run ID:** sud23155
**Status:** Finished (per W&B)

### Training Loss Curve
![Training Loss](images/phase2_training_loss.pdf)

### Eval Loss Curve
![Eval Loss](images/phase2_eval_loss.pdf)

### Final Metrics
- Final Training Loss: 1.0949
- Final Eval Loss: 0.8840
- Steps: 500/500
- Peak VRAM: ~9GB / 24GB
- GPU Utilization: ~100%
- Training Time: ~40 minutes

## Checklist (Artifacts)
- [x] Training loss chart: docs/images/phase2_training_loss.pdf
- [x] Eval loss chart: docs/images/phase2_eval_loss.pdf
- [x] Optional GPU metrics: docs/images/GPUmetrics.png

**Status:** Complete (PDFs attached)
**Date:** 2026-01-17
**Duration:** ~40 minutes

---

## Training Configuration

### Model
- Base Model: Qwen2.5-Coder-7B-Instruct
- Method: QLoRA (Parameter-Efficient Fine-Tuning)
- Quantization: 4-bit (NF4)

### LoRA Parameters
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target Modules: q_proj, v_proj, k_proj, o_proj

### Training Hyperparameters
- Learning Rate: 2e-4
- Optimizer: AdamW (8-bit)
- Scheduler: Cosine with warmup
- Batch Size: 4 (effective, via gradient accumulation)
- Max Steps: 500
- Gradient Checkpointing: Enabled
- Mixed Precision: bfloat16

### Dataset
- Training Examples: 1,388
- Validation Examples: 173
- Test Examples: 174
- Preprocessing: Phase 1 pipeline (97.71% shellcheck pass, 0 dangerous commands)

---

## Training Results

### Final Metrics
- Final Training Loss: 1.0949
- Final Eval Loss: 0.8840
- Steps Completed: 500/500
- Convergence: Loss decreased from ~3.7 to ~1.09 (per W&B)

### System Performance
- GPU: NVIDIA GPU (RunPod)
- Peak VRAM Usage: ~9GB / 24GB (38% utilization)
- GPU Utilization: ~100% (optimal)
- Power Usage: ~230W (stable)

---

## Checkpoint Information

### Local Checkpoint Location
- models/checkpoints/phase2-final/

### Checkpoint Contents
```
phase2-final/
|-- adapter_config.json
|-- adapter_model.safetensors
|-- tokenizer.json
|-- vocab.json
|-- config.json
|-- chat_template.jinja
`-- README.md
```

### Model Size
- Base Model (4-bit): ~3.5GB
- LoRA Adapters: ~20MB
- Total Download Size: ~20MB (adapters only)

---

## Provenance

### Training Environment
- Platform: RunPod
- Date: 2026-01-17
- Training Framework: Axolotl
- Tracking: Weights & Biases

### Reproducibility
- Config File: configs/axolotl_config.yaml
- Dataset Hash: data/processed/provenance.json
- Random Seed: See Axolotl config
- W&B Run: sud23155 (full hyperparameters logged)

---

## Next Steps

### Phase 3: Evaluation (Upcoming)
1. Domain-Specific Metrics:
   - Exact match accuracy on test set
   - Command-only rate (no explanations)
   - Syntax validation with shellcheck

2. Safety Validation:
   - Test against dangerous command patterns
   - Verify zero dangerous outputs
   - CommandRisk integration testing

3. General Capability Retention:
   - GSM8K (math reasoning)
   - HumanEval (code generation)
   - Compare to base model

4. Human Evaluation:
   - Manual review of 50 random generations
   - Assess command quality and correctness
   - Identify edge cases

---

## Related Files

- Training config: configs/axolotl_config.yaml
- Preprocessing results: data/processed/provenance.json
- W&B run: https://wandb.ai/mwill-itmission20/cli-tuner/runs/sud23155
- Local checkpoint: models/checkpoints/phase2-final/
