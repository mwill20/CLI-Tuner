# Phase 2 - Training: QLoRA Fine-Tuning and Validation

## 1. Introduction

### Learning objectives
- Explain why QLoRA and 4-bit NF4 quantization are used for 7B training
- Read and verify the Axolotl training config used for Phase 2
- Understand the smoke test config and why it exists
- Interpret the Phase 2 training results and W&B run metadata
- Validate a checkpoint using the post-training inference script
- Locate Phase 2 artifacts (checkpoint, results doc, provenance)

### Plain-English explanation
Phase 2 takes the safe dataset from Phase 1 and fine-tunes Qwen2.5-Coder with QLoRA. The goal is to train a lightweight adapter that learns the task without needing full model weights. Think of it as teaching a specialist module that plugs into the base model.

### Why this matters
Training is where the model actually learns the task. If the config is wrong or results are not captured, you cannot reproduce the run or trust the model behavior. This phase establishes reproducible training and verifiable artifacts.

---

## 2. Key Concepts

### Domain terminology
- QLoRA: 4-bit quantized LoRA fine-tuning for large models on limited VRAM.
- LoRA adapter: Small trainable weights that modify attention layers without changing the base model.
- NF4 quantization: 4-bit NormalFloat format for efficient storage and compute.
- Gradient accumulation: Simulates larger batches by accumulating gradients over multiple micro-batches.
- Gradient checkpointing: Saves memory by recomputing activations during backprop.
- W&B tracking: Experiment metadata, losses, and system metrics logged to Weights & Biases.

### Design decisions
- Use QLoRA (4-bit) to fit a 7B model on 24GB GPUs.
- Train only attention projections (q/v/k/o) for high leverage with minimal parameters.
- Use a separate validation set (`data/processed/val.jsonl`) instead of splitting train.
- Fix steps at 500 for a bounded run with reproducible metrics.
- Save only a small number of checkpoints to avoid disk bloat.

### Architecture context
Phase 2 is the training step between Phase 1 data pipeline and Phase 3 evaluation. Inference-time guardrails are not implemented yet.

---

## 3. Code Walkthrough

### File: `configs/axolotl_config.yaml`

#### Base model and quantization (lines 8-18)
These settings define the base model and enable 4-bit QLoRA.

Lines 8-18:
```yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: false

load_in_4bit: true
adapter: lora
```

#### LoRA configuration (lines 22-32)
Rank, alpha, and target modules control what is trained.

Lines 22-32:
```yaml
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
```

#### Dataset and validation (lines 37-48)
Training uses `train.jsonl` and validation uses `val.jsonl`.

Lines 37-48:
```yaml
datasets:
  - path: data/processed/train.jsonl
    type: completion
    field: text

val_set_size: 0.0  # Don't split from train - use test_datasets instead

test_datasets:
  - path: data/processed/val.jsonl
    type: completion
    field: text
```

#### Training and checkpointing (lines 69-94)
These settings define the step count and checkpoint cadence.

Lines 69-94:
```yaml
num_epochs: 1
max_steps: 500  # ~1.4 epochs for 1,388 train examples

optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002
warmup_steps: 50  # 10% of max_steps

bf16: true
fp16: false
gradient_checkpointing: true

output_dir: models/checkpoints
save_strategy: steps
save_steps: 100
save_total_limit: 3
```

#### W&B settings (lines 110-112)
This run logs to the `cli-tuner` project without uploading the model.

Lines 110-112:
```yaml
wandb_project: cli-tuner
wandb_run_name: phase2-training-v1
wandb_log_model: "false"
```

---

### File: `configs/axolotl_smoke_test.yaml`

#### Smoke test datasets and steps (lines 36-68)
The smoke test uses small files and just 10 steps.

Lines 36-68:
```yaml
datasets:
  - path: data/processed/train_smoke.jsonl
    type: completion
    field: text

val_set_size: 0.0

test_datasets:
  - path: data/processed/val_smoke.jsonl
    ds_type: json
    split: train
    type: completion

num_epochs: 1
max_steps: 10  # Just 10 steps to verify setup
```

---

### File: `scripts/validate_checkpoint.py`

#### CLI arguments and defaults (lines 32-109)
The script defaults to the Phase 2 checkpoint and uses test.jsonl for sampling.

Lines 32-109:
```python
parser.add_argument(
    "--checkpoint-dir",
    default=str(ROOT_DIR / "models" / "checkpoints" / "phase2-final"),
    help="Path to the LoRA checkpoint directory.",
)
parser.add_argument(
    "--data-path",
    default=str(PROCESSED_DIR / "test.jsonl"),
    help="JSONL dataset to sample prompts from.",
)
parser.add_argument(
    "--sample-size",
    type=int,
    default=5,
    help="Number of samples to generate.",
)
parser.add_argument(
    "--load-in-4bit",
    dest="load_in_4bit",
    action="store_true",
    help="Load the base model in 4-bit (requires CUDA + bitsandbytes).",
)
parser.set_defaults(load_in_4bit=True)
```

#### Model loading and adapter merge (lines 148-191)
The base model is loaded and the LoRA adapter is applied.

Lines 148-191:
```python
has_cuda = torch.cuda.is_available()
if not has_cuda and not allow_cpu:
    raise RuntimeError("CUDA not available. Use --allow-cpu to attempt CPU inference.")

torch_dtype = torch.bfloat16 if has_cuda else torch.float32
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map=device_map,
    load_in_4bit=load_in_4bit,
    torch_dtype=torch_dtype,
    trust_remote_code=False,
)
model = PeftModel.from_pretrained(base, str(checkpoint_dir))
model.eval()
```

#### Generation and safety check (lines 194-246)
Each generated output is checked against the dangerous command patterns.

Lines 194-246:
```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(**inputs, **generation_kwargs)

gen_ids = generated[0][inputs["input_ids"].shape[1] :]
text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
dangerous = is_dangerous_command(text)
```

#### W&B validation run (lines 249-303)
Validation outputs are logged, and the run is closed.

Lines 249-303:
```python
wandb.init(project="ready-tensor-llm", name="validation_run_500")
...
if args.output_jsonl:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in generated:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
...
wandb.finish()
```

---

## 4. Hands-On Exercises

### Prerequisites
- Linux/WSL2 or RunPod GPU for training
- W&B API key configured in `.env`
- Phase 1 outputs present in `data/processed/`
- Phase 2 checkpoint present at `models/checkpoints/phase2-final/`

### Exercise 1: Verify checkpoint contents

```bash
# Bash
ls -1 models/checkpoints/phase2-final/
```

Expected output (file names):
```text
adapter_config.json
adapter_model.safetensors
added_tokens.json
chat_template.jinja
config.json
merges.txt
README.md
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
```

### Exercise 2: Run post-training validation

```bash
python scripts/validate_checkpoint.py \
  --checkpoint-dir models/checkpoints/phase2-final \
  --sample-size 5 \
  --output-jsonl data/validation/phase2_results.jsonl
```

Expected output (structure only):
```text
INFO Sample 1
INFO Instruction: <text>
INFO Generated: <command>
```

### Exercise 3: Review training results

Open:
- `docs/phase2_training_results.md`
- `docs/images/phase2_training_loss.pdf`
- `docs/images/phase2_eval_loss.pdf`

Confirm metrics:
- Final Training Loss: 1.0949
- Final Eval Loss: 0.8840
- Steps: 500/500

### Exercise 4: Verify Phase 2 provenance

```bash
cat models/checkpoints/phase2-final/provenance.json
```

Expected fields:
```json
{
  "phase": "Phase 2 - Training",
  "base_model": "Qwen2.5-Coder-7B-Instruct",
  "method": "QLoRA",
  "final_metrics": {
    "train_loss": 1.0949,
    "eval_loss": 0.884,
    "steps": 500
  },
  "checkpoint": "models/checkpoints/phase2-final/"
}
```

Common pitfalls:
- Running validation without CUDA: use `--allow-cpu` (slow) or run on GPU.
- W&B not logging: confirm `WANDB_API_KEY` in `.env`.
- Dangerous outputs: a flagged output means Phase 3 guardrails are still required.

---

## 5. Interview Preparation

### Question 1
"Why use QLoRA instead of full fine-tuning?"

Model answer:
"QLoRA lets me fine-tune a 7B model on a 24GB GPU by quantizing the base model to 4-bit and training only low-rank adapters. This reduces memory and compute while preserving performance. It is the only practical option for single-GPU training without sacrificing the base model weights."

### Question 2
"Why target q_proj, k_proj, v_proj, and o_proj?"

Model answer:
"Those are the attention projection layers where small weight updates have high impact. Training only these layers keeps the adapter small and efficient while still capturing task-specific behavior."

### Question 3
"How do you ensure the validation set is truly held out?"

Model answer:
"Phase 1 produced separate train/val/test splits with deduplication. The Axolotl config uses `val.jsonl` as a dedicated validation dataset, so no training examples leak into eval."

### Question 4
"What does your post-training validation script check?"

Model answer:
"It loads the base model with the LoRA adapter, samples prompts from `test.jsonl`, generates outputs, and runs a dangerous-command check on each output. It also logs the run to W&B and can export the samples as JSONL."

### Question 5
"What are the key success criteria for Phase 2?"

Model answer:
"Training completes without OOM, losses converge (train 1.0949, eval 0.8840 at step 500), the checkpoint is saved to `models/checkpoints/phase2-final/`, and artifacts are documented with W&B and provenance."

---

## 6. Key Takeaways

- QLoRA makes 7B fine-tuning feasible on 24GB GPUs
- Training uses LoRA on attention projections only
- Axolotl config defines reproducible steps, batch size, and eval cadence
- Validation uses held-out test prompts and flags dangerous outputs
- Phase 2 artifacts include checkpoint, results doc, and provenance

---

## 7. Summary Reference Card

### Inputs
- `data/processed/train.jsonl` (1,388 examples)
- `data/processed/val.jsonl` (173 examples)
- `data/processed/test.jsonl` (174 examples)

### Outputs
- `models/checkpoints/phase2-final/` (LoRA adapter checkpoint)
- `docs/phase2_training_results.md` (training report)
- `docs/images/phase2_training_loss.pdf`
- `docs/images/phase2_eval_loss.pdf`
- `models/checkpoints/phase2-final/provenance.json`
- `data/validation/phase2_results.jsonl` (sample outputs)

### Key commands
- `accelerate launch -m axolotl.cli.train configs/axolotl_config.yaml`
- `accelerate launch -m axolotl.cli.train configs/axolotl_smoke_test.yaml`
- `python scripts/validate_checkpoint.py --checkpoint-dir models/checkpoints/phase2-final --sample-size 5`

### Results
- W&B run: https://wandb.ai/mwill-itmission20/cli-tuner/runs/sud23155
- Final training loss: 1.0949
- Final eval loss: 0.8840
- Steps: 500

---

## 8. Next Steps

- Phase 3: Evaluation (exact match, command-only rate, safety validation)
- Add inference-time guardrails for dangerous commands
- Expand evaluation set and compare to base model

---

## 9. General Best Practices (Not All Implemented Here)

### Safety and validation
- Implemented: training-time safety filtering (Phase 1)
- Implemented: post-training output checks in `validate_checkpoint.py`
- Not implemented: inference-time guardrails (Phase 3)

### Reproducibility
- Implemented: fixed training config in `configs/axolotl_config.yaml`
- Implemented: W&B run logging
- Implemented: checkpoint provenance in `models/checkpoints/phase2-final/provenance.json`
