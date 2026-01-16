"""
Phase 1 data pipeline: load, validate, filter, template, tokenize, split, and record provenance.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from datasets import load_dataset
from transformers import AutoTokenizer
from pydantic import ValidationError

from guardrails.patterns import ZERO_TOLERANCE_PATTERNS, is_dangerous_command
from logging_utils import configure_logging
from schemas.dataset import BashCommandExample


BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET_ID = "prabhanshubhowal/natural_language_to_linux"
SHELLCHECK_TIMEOUT = 5
MAX_LENGTH = 2048
MIN_DATASET_SIZE = 500
SPLIT_SEED = 42
SAMPLE_SIZE = 1835  # Set to integer (e.g., 2000) to process a random sample for testing
SAMPLE_SEED = 42  # Random seed for sampling when SAMPLE_SIZE is set

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
PROCESSED_DIR = DATA_DIR / "processed"
RUNTIME_LOG_DIR = ROOT_DIR / "logs"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for debug logging."""
    parser = argparse.ArgumentParser(
        description="Phase 1 data pipeline: load, validate, filter, template, tokenize, split, and record provenance."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to logs/ with timestamped filenames.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    """Ensure required directories exist."""
    for path in [DATA_DIR / "raw", LOG_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_jsonl(path: Path, records: list[dict]) -> None:
    """Write list of dicts to JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_and_validate_dataset(dataset_id: str) -> dict:
    """
    Load dataset from HuggingFace Hub and validate schema.

    Returns:
        dict with keys: valid_examples, schema_violations, missing_fields
    """
    attempts = 3
    dataset = None
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            dataset = load_dataset(dataset_id, trust_remote_code=False)
            break
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            backoff = 2 ** (attempt - 1)
            LOGGER.warning(
                "Download failed (attempt %s/%s), retrying in %ss...",
                attempt,
                attempts,
                backoff,
            )
            time.sleep(backoff)

    if dataset is None:
        raise RuntimeError(
            f"Failed to download dataset after {attempts} retries: {last_error}"
        )

    if hasattr(dataset, "keys"):
        if "train" in dataset:
            records = dataset["train"]
        else:
            records = dataset[next(iter(dataset.keys()))]
    else:
        records = dataset

    total_downloaded = len(records)
    
    # Sample dataset if SAMPLE_SIZE is set
    if SAMPLE_SIZE is not None and len(records) > SAMPLE_SIZE:
        rng = random.Random(SAMPLE_SEED)
        indices = rng.sample(range(len(records)), SAMPLE_SIZE)
        indices.sort()
        LOGGER.info(
            "  Sampling %s examples from %s total (seed=%s)",
            SAMPLE_SIZE,
            total_downloaded,
            SAMPLE_SEED,
        )
        if hasattr(records, "select"):
            records = records.select(indices)
        else:
            records = [records[i] for i in indices]
    
    valid_examples: list[dict] = []
    schema_violations: list[dict] = []
    missing_fields: list[dict] = []

    # Field mapping: dataset uses nl_command/bash_code, we need instruction/output
    field_mapping = {
        "nl_command": "instruction",
        "bash_code": "output",
    }

    for idx, record in enumerate(records):
        # Normalize field names
        normalized_record = {}
        for old_field, new_field in field_mapping.items():
            if old_field in record:
                normalized_record[new_field] = record[old_field]
        
        # Check for input field (optional, defaults to empty string)
        normalized_record["input"] = record.get("input", "")

        # Check for required fields
        required_fields = ["instruction", "output"]
        missing = [field for field in required_fields if field not in normalized_record or not normalized_record[field]]
        if missing:
            for field in missing:
                missing_fields.append({"index": idx, "missing_field": field})
            continue

        try:
            validated = BashCommandExample(**normalized_record)
            if hasattr(validated, "model_dump"):
                valid_examples.append(validated.model_dump())
            else:
                valid_examples.append(validated.dict())
        except ValidationError as exc:
            schema_violations.append(
                {
                    "index": idx,
                    "instruction": normalized_record.get("instruction"),
                    "input": normalized_record.get("input", ""),
                    "output": normalized_record.get("output"),
                    "violation_reason": str(exc),
                }
            )

    save_jsonl(LOG_DIR / "schema_violations.jsonl", schema_violations)
    save_jsonl(LOG_DIR / "missing_fields.jsonl", missing_fields)

    return {
        "total_downloaded": total_downloaded,
        "valid_examples": valid_examples,
        "schema_violations": schema_violations,
        "missing_fields": missing_fields,
    }


def parse_shellcheck_version(version_output: str) -> tuple[int, int, int] | None:
    """Extract a semantic version tuple from shellcheck --version output."""
    for line in version_output.splitlines():
        line = line.strip()
        if line.startswith("version:"):
            version_str = line.split("version:", 1)[1].strip()
            parts = version_str.split(".")
            try:
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                patch = int(parts[2]) if len(parts) > 2 else 0
            except ValueError:
                return None
            return (major, minor, patch)
    return None


def check_shellcheck_installed() -> str:
    """Verify shellcheck is installed and get version."""
    try:
        result = subprocess.run(
            ["shellcheck", "--version"],
            capture_output=True,
            timeout=SHELLCHECK_TIMEOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Shellcheck not installed. Install: apt-get install shellcheck (Linux) "
            "or brew install shellcheck (Mac)"
        ) from exc

    version_output = result.stdout.decode("utf-8", errors="ignore")
    version_tuple = parse_shellcheck_version(version_output)
    if version_tuple is None:
        raise RuntimeError("Unable to parse shellcheck version output")
    if version_tuple < (0, 9, 0):
        raise RuntimeError(f"Shellcheck version too old: {version_output.strip()}")
    return version_output.strip()


def validate_bash_syntax(command: str) -> dict:
    """
    Validate Bash command syntax using shellcheck.

    Returns:
        dict with keys: valid (bool), errors (list)
    """
    try:
        result = subprocess.run(
            ["shellcheck", "--shell=bash", "--severity=error", "--format=json", "-"],
            input=command.encode("utf-8"),
            capture_output=True,
            timeout=SHELLCHECK_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"valid": False, "errors": [{"message": "Shellcheck timeout"}]}
    except FileNotFoundError as exc:
        raise RuntimeError("Shellcheck not installed. Install: apt-get install shellcheck") from exc

    if result.returncode == 0:
        return {"valid": True, "errors": []}

    stdout = result.stdout.decode("utf-8", errors="ignore").strip()
    if not stdout:
        return {"valid": False, "errors": [{"message": "Shellcheck error"}]}
    try:
        errors = json.loads(stdout)
    except json.JSONDecodeError:
        errors = [{"message": stdout}]
    return {"valid": False, "errors": errors}


def filter_invalid_syntax(examples: list[dict], shellcheck_version: str | None = None) -> dict:
    """
    Filter examples with invalid syntax.

    Returns:
        dict with keys: valid_examples, removed_examples
    """
    valid_examples: list[dict] = []
    removed_examples: list[dict] = []
    
    total = len(examples)
    LOGGER.info("  Validating %s commands with shellcheck...", total)

    for i, example in enumerate(examples):
        if (i + 1) % 100 == 0:
            LOGGER.info("  Progress: %s/%s (%s%%)", i + 1, total, 100 * (i + 1) // total)
        
        command = example["output"]
        validation = validate_bash_syntax(command)

        if validation["valid"]:
            valid_examples.append(example)
        else:
            removed_examples.append(
                {
                    "instruction": example["instruction"],
                    "output": command,
                    "shellcheck_errors": validation["errors"],
                }
            )

    LOGGER.info(
        "  Shellcheck complete: %s/%s passed (%s%%)",
        len(valid_examples),
        total,
        100 * len(valid_examples) // total,
    )
    save_jsonl(LOG_DIR / "removed_invalid_syntax.jsonl", removed_examples)

    return {
        "valid_examples": valid_examples,
        "removed_examples": removed_examples,
        "shellcheck_version": shellcheck_version,
    }


def filter_dangerous_commands(examples: list[dict]) -> dict:
    """
    Filter examples with dangerous commands.

    Returns:
        dict with keys: safe_examples, removed_examples, patterns_matched
    """
    safe_examples: list[dict] = []
    removed_examples: list[dict] = []
    patterns_matched: dict[str, int] = {}

    for example in examples:
        command = example["output"]
        check = is_dangerous_command(command)

        if not check["is_dangerous"]:
            safe_examples.append(example)
        else:
            removed_examples.append(
                {
                    "instruction": example["instruction"],
                    "output": command,
                    "pattern_matched": check["pattern_matched"],
                }
            )
            patterns_matched[check["pattern_matched"]] = (
                patterns_matched.get(check["pattern_matched"], 0) + 1
            )

    save_jsonl(LOG_DIR / "removed_dangerous.jsonl", removed_examples)

    for example in safe_examples:
        check = is_dangerous_command(example["output"])
        if check["is_dangerous"]:
            raise RuntimeError(f"Dangerous command slipped through: {example['output']}")

    return {
        "safe_examples": safe_examples,
        "removed_examples": removed_examples,
        "patterns_matched": patterns_matched,
    }


def verify_no_dangerous_commands(examples: list[dict]) -> None:
    """
    CRITICAL: Assert no dangerous commands remain before saving.
    """
    for example in examples:
        check = is_dangerous_command(example["output"])
        if check["is_dangerous"]:
            raise RuntimeError(
                f"CRITICAL: Dangerous command slipped through: {example['output']}"
            )


def apply_chat_template(examples: list[dict]) -> list[dict]:
    """
    Apply Qwen2.5 chat template to all examples.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)

    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        raise RuntimeError("Tokenizer does not have chat_template attribute")

    formatted_examples: list[dict] = []

    for example in examples:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Verify format (Qwen uses <|im_start|> and <|im_end|>)
        if "<|im_start|>" not in formatted_text:
            raise RuntimeError("im_start token missing from formatted text")
        if "<|im_end|>" not in formatted_text:
            raise RuntimeError("im_end token missing from formatted text")
        if "user" not in formatted_text or "assistant" not in formatted_text:
            raise RuntimeError("User/assistant roles missing from formatted text")
        if "{instruction}" in formatted_text or "{output}" in formatted_text:
            raise RuntimeError("Template placeholders leaked into formatted text")

        formatted_examples.append(
            {
                "instruction": example["instruction"],
                "output": example["output"],
                "text": formatted_text,
            }
        )

    return formatted_examples


def tokenize_with_masking(examples: list[dict], tokenizer) -> list[dict]:
    """
    Tokenize examples and apply assistant-only masking.
    """
    tokenized_examples: list[dict] = []

    # Qwen uses "assistant\n" after <|im_start|> to mark assistant responses.

    for example in examples:
        full_encoding = tokenizer(
            example["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors=None,
        )

        input_ids = full_encoding["input_ids"]
        labels = input_ids.copy()

        # Find where assistant response starts
        # Look for <|im_start|>assistant pattern in the text
        text = example["text"]
        assistant_marker = "<|im_start|>assistant\n"
        assistant_pos = text.find(assistant_marker)
        
        if assistant_pos == -1:
            raise RuntimeError(f"Assistant marker not found in: {text[:100]}")
        
        # Tokenize up to assistant response to find the split point
        pre_assistant = text[:assistant_pos + len(assistant_marker)]
        pre_tokens = tokenizer(pre_assistant, return_tensors=None)["input_ids"]
        assistant_start_idx = len(pre_tokens)

        # Mask user tokens (everything before assistant response)
        labels[:assistant_start_idx] = [-100] * assistant_start_idx

        tokenized_examples.append(
            {
                "instruction": example["instruction"],
                "output": example["output"],
                "text": example["text"],
                "input_ids": input_ids,
                "labels": labels,
            }
        )

    return tokenized_examples


def verify_masking(tokenized_examples: list[dict], tokenizer) -> bool:
    """Verify assistant-only masking is correct."""
    
    for example in tokenized_examples:
        labels = example["labels"]

        # Check that some tokens are masked (-100) and some are not
        masked_count = sum(1 for label in labels if label == -100)
        unmasked_count = sum(1 for label in labels if label != -100)
        
        if masked_count == 0:
            raise RuntimeError("No user tokens were masked")
        if unmasked_count == 0:
            raise RuntimeError("All tokens were masked - assistant should not be masked")

    return True


def record_fingerprint(example: dict) -> str:
    """Compute a stable fingerprint for a single example."""
    payload = json.dumps(
        {
            "instruction": example["instruction"],
            "input": example.get("input", ""),
            "output": example["output"],
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_dataset(examples: list[dict], seed: int = SPLIT_SEED) -> dict:
    """
    Split dataset into train/val/test.
    """
    # First, deduplicate examples using fingerprints
    seen_fingerprints = set()
    unique_examples = []
    
    for example in examples:
        fp = record_fingerprint(example)
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            unique_examples.append(example)
    
    duplicates_removed = len(examples) - len(unique_examples)
    if duplicates_removed > 0:
        LOGGER.info("  Removed %s duplicate examples", duplicates_removed)
    
    examples = unique_examples
    
    random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    train = examples[:train_size]
    val = examples[train_size : train_size + val_size]
    test = examples[train_size + val_size :]

    # Verify no leakage (should be impossible now after deduplication)
    train_fingerprints = {record_fingerprint(ex) for ex in train}
    val_fingerprints = {record_fingerprint(ex) for ex in val}
    test_fingerprints = {record_fingerprint(ex) for ex in test}

    if train_fingerprints & val_fingerprints:
        raise RuntimeError("Train-Val leakage detected")
    if train_fingerprints & test_fingerprints:
        raise RuntimeError("Train-Test leakage detected")
    if val_fingerprints & test_fingerprints:
        raise RuntimeError("Val-Test leakage detected")

    return {"train": train, "val": val, "test": test}


def compute_dataset_hash(examples: list[dict]) -> str:
    """Compute SHA256 hash of dataset for provenance."""
    data_str = json.dumps(examples, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def record_provenance(stats: dict, splits: dict, output_path: Path) -> None:
    """
    Record full data provenance.
    """
    provenance = {
        "source_dataset": DATASET_ID,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "collection_date": datetime.now(timezone.utc).isoformat(),
        "total_examples_downloaded": stats["total_downloaded"],
        "schema_violations": stats["schema_violations"],
        "missing_fields": stats["missing_fields"],
        "invalid_syntax": stats["invalid_syntax"],
        "dangerous_commands": stats["dangerous_commands"],
        "final_train_size": len(splits["train"]),
        "final_val_size": len(splits["val"]),
        "final_test_size": len(splits["test"]),
        "shellcheck_version": stats["shellcheck_version"],
        "shellcheck_pass_rate": stats["shellcheck_pass_rate"],
        "data_hash_sha256": compute_dataset_hash(splits["train"]),
        "filtering_config": {
            "zero_tolerance_patterns": ZERO_TOLERANCE_PATTERNS,
            "chat_template_source": f"{BASE_MODEL} tokenizer.chat_template",
            "masking_strategy": "assistant_only (user tokens = -100)",
            "sample_size": SAMPLE_SIZE,
            "sample_seed": SAMPLE_SEED,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=True)


def main() -> None:
    """Execute full 7-component pipeline."""
    ensure_directories()

    stats: dict = {}

    args = parse_args()
    log_file = configure_logging(RUNTIME_LOG_DIR, args.debug, "preprocess_data")
    if log_file:
        LOGGER.info("Debug log file: %s", log_file)

    LOGGER.info("Step 1/7: Loading dataset...")
    data = load_and_validate_dataset(DATASET_ID)
    stats["total_downloaded"] = data["total_downloaded"]
    stats["schema_violations"] = len(data["schema_violations"])
    stats["missing_fields"] = len(data["missing_fields"])

    LOGGER.info("Step 2/7: Running shellcheck...")
    stats["shellcheck_version"] = check_shellcheck_installed()
    valid_data = filter_invalid_syntax(data["valid_examples"], stats["shellcheck_version"])
    stats["invalid_syntax"] = len(valid_data["removed_examples"])
    total_before_shellcheck = len(data["valid_examples"])
    if total_before_shellcheck:
        stats["shellcheck_pass_rate"] = round(
            (len(valid_data["valid_examples"]) / total_before_shellcheck) * 100.0, 2
        )
    else:
        stats["shellcheck_pass_rate"] = 0.0

    LOGGER.info("Step 3/7: Filtering dangerous commands...")
    safe_data = filter_dangerous_commands(valid_data["valid_examples"])
    stats["dangerous_commands"] = len(safe_data["removed_examples"])
    stats["patterns_matched"] = safe_data["patterns_matched"]

    if len(safe_data["safe_examples"]) < MIN_DATASET_SIZE:
        LOGGER.warning(
            "Dataset size below %s after filtering (%s examples)",
            MIN_DATASET_SIZE,
            len(safe_data["safe_examples"]),
        )

    LOGGER.info("Step 4/7: Applying chat template...")
    formatted_data = apply_chat_template(safe_data["safe_examples"])

    LOGGER.info("Step 5/7: Tokenizing with assistant-only masking...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    tokenized_data = tokenize_with_masking(formatted_data, tokenizer)
    verify_masking(tokenized_data, tokenizer)

    LOGGER.info("Step 6/7: Splitting train/val/test...")
    splits = split_dataset(tokenized_data, seed=SPLIT_SEED)
    stats["final_train"] = len(splits["train"])
    stats["final_val"] = len(splits["val"])
    stats["final_test"] = len(splits["test"])

    verify_no_dangerous_commands(splits["train"] + splits["val"] + splits["test"])

    LOGGER.info("Step 7/7: Saving outputs...")
    save_jsonl(PROCESSED_DIR / "train.jsonl", splits["train"])
    save_jsonl(PROCESSED_DIR / "val.jsonl", splits["val"])
    save_jsonl(PROCESSED_DIR / "test.jsonl", splits["test"])
    record_provenance(stats, splits, PROCESSED_DIR / "provenance.json")

    LOGGER.info(
        "Pipeline complete. %s train, %s val, %s test.",
        stats["final_train"],
        stats["final_val"],
        stats["final_test"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Pipeline failed")
        raise
