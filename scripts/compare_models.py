"""
Phase 3 comparison: evaluate base vs fine-tuned models.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.eval_utils import (
    is_command_only,
    load_jsonl,
    normalize_command,
    save_json,
    validate_bash_syntax,
)
from scripts.logging_utils import configure_logging
from schemas.evaluation import EvaluationConfig, GenerationConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_LOG_DIR = ROOT_DIR / "logs"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare base model vs fine-tuned checkpoint on domain metrics."
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
        "--output-dir",
        required=True,
        help="Directory to write comparison outputs.",
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


def load_base_model(base_model: str, load_in_4bit: bool, device_map: str) -> tuple:
    """Load base model without adapters."""
    try:
        import torch
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
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=False,
    )
    return model, tokenizer


def load_finetuned_model(
    base_model: str,
    checkpoint_dir: Path,
    load_in_4bit: bool,
    device_map: str,
) -> tuple:
    """Load base model and apply LoRA adapters."""
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


def run_comparative_evaluation(
    records: list[dict],
    base_model,
    base_tokenizer,
    finetuned_model,
    finetuned_tokenizer,
    generation,
) -> dict:
    """Evaluate base and fine-tuned models on domain metrics."""
    base_metrics = evaluate_domain_metrics(
        records, base_model, base_tokenizer, generation
    )
    finetuned_metrics = evaluate_domain_metrics(
        records, finetuned_model, finetuned_tokenizer, generation
    )

    improvement = finetuned_metrics.get("exact_match_accuracy", 0) - base_metrics.get(
        "exact_match_accuracy", 0
    )
    return {
        "base": base_metrics,
        "finetuned": finetuned_metrics,
        "improvements": {"exact_match": round(improvement, 4)},
    }


def generate_command(
    model,
    tokenizer,
    record: dict,
    generation: GenerationConfig,
) -> str:
    """Generate a command from a single record."""
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

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)

    gen_ids = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_domain_metrics(
    records: list[dict],
    model,
    tokenizer,
    generation: GenerationConfig,
) -> dict:
    """Compute domain metrics for a model."""
    predictions: list[str] = []
    references: list[str] = []
    syntax_results: list[bool] = []

    for record in records:
        instruction = record.get("instruction", "").strip()
        reference = record.get("output", "").strip()
        if not instruction or not reference:
            continue

        prediction = generate_command(model, tokenizer, record, generation)
        predictions.append(prediction)
        references.append(reference)

        syntax_verdict = validate_bash_syntax(prediction)
        if syntax_verdict is not None:
            syntax_results.append(syntax_verdict)

    total = len(predictions)
    exact_matches = sum(
        1 for predicted, reference in zip(predictions, references)
        if normalize_command(predicted) == normalize_command(reference)
    )
    command_only_count = sum(1 for output in predictions if is_command_only(output))
    syntax_rate = None
    if syntax_results:
        syntax_rate = round(
            sum(1 for verdict in syntax_results if verdict) / len(syntax_results), 4
        )

    return {
        "total_examples": total,
        "exact_match_accuracy": round(exact_matches / total, 4) if total else 0.0,
        "command_only_rate": round(command_only_count / total, 4) if total else 0.0,
        "syntax_validity": syntax_rate,
        "successful_generations": total,
        "failed_generations": 0,
    }


def generate_comparison_report(metrics: dict) -> str:
    """Generate a markdown report comparing metrics."""
    base = metrics.get("base", {})
    finetuned = metrics.get("finetuned", {})
    lines = [
        "# Phase 3 Model Comparison",
        "",
        "## Summary",
        "",
        "| Metric | Base | Fine-tuned |",
        "| --- | --- | --- |",
        f"| Exact match | {base.get('exact_match_accuracy')} | {finetuned.get('exact_match_accuracy')} |",
        f"| Command-only rate | {base.get('command_only_rate')} | {finetuned.get('command_only_rate')} |",
        f"| Syntax validity | {base.get('syntax_validity')} | {finetuned.get('syntax_validity')} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_file = configure_logging(RUNTIME_LOG_DIR, args.debug, "compare_models")
    if log_file:
        LOGGER.info("Debug log file: %s", log_file)

    config = EvaluationConfig.from_yaml(args.config)
    checkpoint = Path(args.checkpoint)
    test_data = Path(args.test_data)
    output_dir = Path(args.output_dir)

    if not test_data.exists():
        raise RuntimeError(f"Test dataset not found: {test_data}")

    records = load_jsonl(test_data)
    if not records:
        raise RuntimeError("No records found in test dataset.")

    load_in_4bit = (
        args.load_in_4bit
        if args.load_in_4bit is not None
        else config.model.load_in_4bit
    )
    generation = (
        config.generation.realistic
        if args.generation_mode == "realistic"
        else config.generation.deterministic
    )

    base_model, base_tokenizer = load_base_model(
        base_model=config.model.base_model,
        load_in_4bit=load_in_4bit,
        device_map=config.model.device_map,
    )
    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        base_model=config.model.base_model,
        checkpoint_dir=checkpoint,
        load_in_4bit=load_in_4bit,
        device_map=config.model.device_map,
    )

    metrics = run_comparative_evaluation(
        records,
        base_model,
        base_tokenizer,
        finetuned_model,
        finetuned_tokenizer,
        generation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "base_vs_finetuned.json", metrics)

    report = generate_comparison_report(metrics)
    report_path = output_dir / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")

    LOGGER.info("Comparison metrics saved to %s", output_dir / "base_vs_finetuned.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Model comparison failed.")
        raise
