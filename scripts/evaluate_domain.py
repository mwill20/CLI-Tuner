"""
Phase 3 domain evaluation: exact match, command-only rate, syntax validity.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from scripts.eval_utils import (
    is_command_only,
    load_jsonl,
    normalize_command,
    save_json,
    save_jsonl,
    validate_bash_syntax as _validate_bash_syntax,
)
from scripts.logging_utils import configure_logging
from schemas.evaluation import EvaluationConfig, GenerationConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_LOG_DIR = ROOT_DIR / "logs"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run domain evaluation on the Phase 2 checkpoint."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "evaluation_config.yaml"),
        help="Path to evaluation configuration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help="Path to JSONL test dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write metrics.json and results.jsonl.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["deterministic", "realistic"],
        default="deterministic",
        help="Generation mode to use.",
    )
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading when CUDA is available.",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to logs/ with timestamped filenames.",
    )
    parser.set_defaults(load_in_4bit=None)
    return parser.parse_args()


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
    record: dict,
    generation: GenerationConfig,
    max_retries: int = 2,
) -> str:
    """Generate command with retry logic."""
    import torch

    instruction = record.get("instruction", "").strip()
    if not instruction:
        raise ValueError("Record missing instruction field.")
    input_text = record.get("input", "").strip()
    user_content = f"{instruction}\n{input_text}" if input_text else instruction

    messages = [{"role": "user", "content": user_content}]
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

    attempt = 0
    while True:
        attempt += 1
        try:
            with torch.no_grad():
                generated = model.generate(**inputs, **generation_kwargs)
            gen_ids = generated[0][inputs["input_ids"].shape[1] :]
            return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            LOGGER.warning("Generation failed (attempt %s): %s", attempt, exc)
            if attempt >= max_retries:
                raise


def check_command_only(output: str) -> bool:
    """Validate command-only output."""
    return is_command_only(output)


def validate_bash_syntax(command: str) -> Optional[bool]:
    """Shellcheck integration with graceful degradation."""
    return _validate_bash_syntax(command)


def evaluate_domain_metrics(
    records: list[dict],
    model,
    tokenizer,
    generation: GenerationConfig,
) -> tuple[list[dict], dict]:
    """Run the domain evaluation loop and compute metrics."""
    predictions: list[str] = []
    references: list[str] = []
    results: list[dict] = []
    syntax_results: list[bool] = []
    failed_generations = 0

    for record in records:
        instruction = record.get("instruction", "").strip()
        reference = record.get("output", "").strip()
        if not instruction or not reference:
            LOGGER.warning("Skipping record missing instruction/output.")
            continue

        try:
            prediction = generate_command(model, tokenizer, record, generation)
        except Exception as exc:
            failed_generations += 1
            results.append(
                {
                    "instruction": instruction,
                    "reference_output": reference,
                    "predicted_output": "",
                    "exact_match": False,
                    "command_only": False,
                    "syntax_valid": None,
                    "error": str(exc),
                }
            )
            continue

        predictions.append(prediction)
        references.append(reference)

        syntax_verdict = validate_bash_syntax(prediction)
        if syntax_verdict is not None:
            syntax_results.append(syntax_verdict)

        results.append(
            {
                "instruction": instruction,
                "reference_output": reference,
                "predicted_output": prediction,
                "exact_match": normalize_command(prediction)
                == normalize_command(reference),
                "command_only": check_command_only(prediction),
                "syntax_valid": syntax_verdict,
            }
        )

    total = len(predictions)
    exact_matches = sum(
        1 for predicted, reference in zip(predictions, references)
        if normalize_command(predicted) == normalize_command(reference)
    )
    command_only_count = sum(1 for output in predictions if check_command_only(output))
    syntax_rate = None
    if syntax_results:
        syntax_rate = round(
            sum(1 for verdict in syntax_results if verdict) / len(syntax_results), 4
        )

    metrics = {
        "total_examples": total + failed_generations,
        "successful_generations": total,
        "failed_generations": failed_generations,
        "exact_match_accuracy": round(exact_matches / total, 4) if total else 0.0,
        "command_only_rate": round(command_only_count / total, 4) if total else 0.0,
        "syntax_validity": syntax_rate,
    }
    return results, metrics


def main() -> None:
    args = parse_args()
    log_file = configure_logging(RUNTIME_LOG_DIR, args.debug, "evaluate_domain")
    if log_file:
        LOGGER.info("Debug log file: %s", log_file)

    config = EvaluationConfig.from_yaml(args.config)

    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(config.model.checkpoint)
    test_data = Path(args.test_data) if args.test_data else Path(config.datasets.test_data)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output.evaluation_dir) / "domain"

    if not test_data.exists():
        raise RuntimeError(f"Test dataset not found: {test_data}")

    generation = (
        config.generation.realistic
        if args.generation_mode == "realistic"
        else config.generation.deterministic
    )
    load_in_4bit = (
        args.load_in_4bit
        if args.load_in_4bit is not None
        else config.model.load_in_4bit
    )

    LOGGER.info("Loading test data: %s", test_data)
    records = load_jsonl(test_data)
    if not records:
        raise RuntimeError("No records found in test dataset.")

    model, tokenizer = load_model_and_adapter(
        base_model=config.model.base_model,
        checkpoint_dir=checkpoint,
        load_in_4bit=load_in_4bit,
        device_map=config.model.device_map,
    )

    results, metrics = evaluate_domain_metrics(records, model, tokenizer, generation)

    save_jsonl(output_dir / "results.jsonl", results)
    save_json(output_dir / "metrics.json", metrics)

    LOGGER.info("Domain metrics saved to %s", output_dir / "metrics.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Domain evaluation failed.")
        raise
