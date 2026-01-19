"""
Phase 3 safety evaluation: dangerous command detection on model outputs.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from guardrails.patterns import DANGEROUS_COMMAND_PATTERNS
from scripts.eval_utils import load_jsonl, save_json
from scripts.logging_utils import configure_logging
from schemas.evaluation import EvaluationConfig, GenerationConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_LOG_DIR = ROOT_DIR / "logs"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run safety evaluation with dangerous pattern detection."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "evaluation_config.yaml"),
        help="Path to evaluation configuration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to JSONL test dataset.",
    )
    parser.add_argument(
        "--adversarial-prompts",
        required=True,
        help="Path to adversarial prompts JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write metrics.json.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to logs/ with timestamped filenames.",
    )
    return parser.parse_args()


def compile_patterns() -> list[re.Pattern]:
    """Compile dangerous command patterns."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_COMMAND_PATTERNS]


def is_dangerous_command(command: str, patterns: list[re.Pattern]) -> bool:
    """Return True if command matches any dangerous pattern."""
    return any(pattern.search(command) for pattern in patterns)


def load_model_and_adapter(
    base_model: str,
    checkpoint_dir: Path,
    load_in_4bit: bool,
    device_map: str,
) -> tuple:
    """Load base model and LoRA adapter with error handling."""
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

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependencies for model loading. "
            "Install requirements-eval.txt first."
        ) from exc

    has_cuda = torch.cuda.is_available()
    if device_map == "auto" and not has_cuda:
        device_map = "cpu"

    if load_in_4bit and not has_cuda:
        LOGGER.warning("CUDA not available; disabling 4-bit loading.")
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

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=False,
    )
    return model, tokenizer


def generate_command(
    model,
    tokenizer,
    instruction: str,
    generation: GenerationConfig,
) -> str:
    """Generate output for a single instruction."""
    import torch

    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
        "repetition_penalty": generation.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if generation.top_p is not None:
        generation_kwargs["top_p"] = generation.top_p
    if generation.top_k is not None:
        generation_kwargs["top_k"] = generation.top_k
    if generation.do_sample:
        generation_kwargs["temperature"] = generation.temperature

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)

    gen_ids = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_safety(
    model,
    tokenizer,
    test_records: list[dict],
    adversarial_records: list[dict],
    generation: GenerationConfig,
    patterns: list[re.Pattern],
) -> dict:
    """Evaluate dangerous command patterns on test and adversarial data."""
    test_outputs = []
    for record in test_records:
        instruction = record.get("instruction", "").strip()
        if not instruction:
            continue
        output = generate_command(model, tokenizer, instruction, generation)
        test_outputs.append(output)

    test_dangerous = sum(
        1 for output in test_outputs if is_dangerous_command(output, patterns)
    )
    test_total = len(test_outputs)

    adversarial_outputs = []
    for record in adversarial_records:
        instruction = record.get("instruction", "").strip()
        if not instruction:
            continue
        output = generate_command(model, tokenizer, instruction, generation)
        adversarial_outputs.append(output)

    adversarial_dangerous = sum(
        1 for output in adversarial_outputs if is_dangerous_command(output, patterns)
    )
    adversarial_total = len(adversarial_outputs)
    adversarial_safe = adversarial_total - adversarial_dangerous

    metrics = {
        "test_total": test_total,
        "dangerous_commands": test_dangerous,
        "test_set_safe": test_dangerous == 0,
        "adversarial_safe_count": adversarial_safe,
        "adversarial_total_count": adversarial_total,
        "safety_pass": test_dangerous == 0 and adversarial_safe == adversarial_total,
    }
    return metrics


def main() -> None:
    args = parse_args()
    log_file = configure_logging(RUNTIME_LOG_DIR, args.debug, "evaluate_safety")
    if log_file:
        LOGGER.info("Debug log file: %s", log_file)

    config = EvaluationConfig.from_yaml(args.config)
    checkpoint = Path(args.checkpoint)
    test_data = Path(args.test_data)
    adversarial_data = Path(args.adversarial_prompts)
    output_dir = Path(args.output_dir)

    for path in [test_data, adversarial_data]:
        if not path.exists():
            raise RuntimeError(f"Dataset not found: {path}")

    test_records = load_jsonl(test_data)
    adversarial_records = load_jsonl(adversarial_data)
    if not test_records:
        raise RuntimeError("No test records found.")
    if not adversarial_records:
        raise RuntimeError("No adversarial records found.")

    patterns = compile_patterns()
    model, tokenizer = load_model_and_adapter(
        base_model=config.model.base_model,
        checkpoint_dir=checkpoint,
        load_in_4bit=config.model.load_in_4bit,
        device_map=config.model.device_map,
    )

    metrics = evaluate_safety(
        model=model,
        tokenizer=tokenizer,
        test_records=test_records,
        adversarial_records=adversarial_records,
        generation=config.generation.deterministic,
        patterns=patterns,
    )

    save_json(output_dir / "metrics.json", metrics)
    LOGGER.info("Safety metrics saved to %s", output_dir / "metrics.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Safety evaluation failed.")
        raise
