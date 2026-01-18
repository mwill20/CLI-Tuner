"""
Post-training validation: load a LoRA checkpoint and generate sample outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

# W&B logging
import wandb

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from guardrails.patterns import is_dangerous_command
from logging_utils import configure_logging


BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNTIME_LOG_DIR = ROOT_DIR / "logs"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for validation run."""
    parser = argparse.ArgumentParser(
        description="Validate a LoRA checkpoint by generating sample outputs."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT_DIR / "models" / "checkpoints" / "final"),
        help="Path to the LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model name or path.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit (requires CUDA + bitsandbytes).",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU inference (slow, may OOM).",
    )
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Optional path to write generated samples as JSONL.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to logs/ with timestamped filenames.",
    )
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    LOGGER.debug("Loaded %s records from %s", len(records), path)
    return records


def select_samples(records: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Select a deterministic sample from records."""
    if sample_size <= 0:
        raise ValueError("sample_size must be >= 1")
    rng = random.Random(seed)
    if sample_size >= len(records):
        return records
    indices = rng.sample(range(len(records)), sample_size)
    return [records[idx] for idx in indices]


def build_user_content(record: dict) -> str:
    """Build user prompt content from a record."""
    instruction = record.get("instruction", "").strip()
    if not instruction:
        raise ValueError("Record missing instruction field")
    input_text = record.get("input", "").strip()
    if input_text:
        return f"{instruction}\n{input_text}"
    return instruction


def load_model(
    base_model: str,
    checkpoint_dir: Path,
    load_in_4bit: bool,
    allow_cpu: bool,
) -> tuple:
    """Load base model, apply LoRA adapters, and return model + tokenizer."""
    if not checkpoint_dir.exists():
        raise RuntimeError(f"Checkpoint directory not found: {checkpoint_dir}")

    adapter_files = [
        checkpoint_dir / "adapter_model.safetensors",
        checkpoint_dir / "adapter_model.bin",
    ]
    if not any(path.exists() for path in adapter_files):
        raise RuntimeError(
            "No adapter model found in checkpoint directory. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu:
        raise RuntimeError("CUDA not available. Use --allow-cpu to attempt CPU inference.")

    device_map = "auto" if has_cuda else "cpu"
    if not has_cuda and load_in_4bit:
        LOGGER.warning("Disabling 4-bit loading on CPU.")
        load_in_4bit = False

    torch_dtype = torch.bfloat16 if has_cuda else torch.float32
    LOGGER.info("Loading base model: %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    )
    LOGGER.info("Loading LoRA adapters from: %s", checkpoint_dir)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    return model, tokenizer


def generate_outputs(
    model,
    tokenizer,
    samples: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    """Generate outputs for a set of samples."""
    outputs: list[dict] = []
    for idx, record in enumerate(samples, start=1):
        user_content = build_user_content(record)
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)

        gen_ids = generated[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        dangerous = is_dangerous_command(text)

        outputs.append(
            {
                "instruction": record.get("instruction", ""),
                "input": record.get("input", ""),
                "reference_output": record.get("output", ""),
                "generated_output": text,
                "dangerous": dangerous,
            }
        )

        LOGGER.info("Sample %s", idx)
        LOGGER.info("Instruction: %s", record.get("instruction", ""))
        LOGGER.info("Generated: %s", text)
        if dangerous:
            LOGGER.warning("Generated output matched dangerous patterns.")

    return outputs


def main() -> None:
    args = parse_args()
    log_file = configure_logging(RUNTIME_LOG_DIR, args.debug, "validate_checkpoint")
    if log_file:
        LOGGER.info("Debug log file: %s", log_file)
    LOGGER.debug("Debug logging enabled")
    LOGGER.info("Checkpoint dir: %s", args.checkpoint_dir)
    LOGGER.info("Base model: %s", args.base_model)
    LOGGER.info("Data path: %s", args.data_path)
    LOGGER.info("Sample size: %s", args.sample_size)
    LOGGER.info("Load in 4-bit: %s", args.load_in_4bit)

    # Initialize W&B run
    wandb.init(project="ready-tensor-llm", name="validation_run_500")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise RuntimeError(f"Data path not found: {data_path}")

    records = load_jsonl(data_path)
    if not records:
        raise RuntimeError("No records found in data path.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = select_samples(records, args.sample_size, args.seed)
    LOGGER.info("Selected %s samples for validation", len(samples))

    model, tokenizer = load_model(
        base_model=args.base_model,
        checkpoint_dir=Path(args.checkpoint_dir),
        load_in_4bit=args.load_in_4bit,
        allow_cpu=args.allow_cpu,
    )

    generated = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in generated:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        LOGGER.info("Wrote validation outputs to %s", output_path)

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Checkpoint validation failed")
        raise
