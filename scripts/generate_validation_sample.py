"""
Generate validation samples for Overseer review.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from logging_utils import configure_logging


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VALIDATION_DIR = DATA_DIR / "validation"
RUNTIME_LOG_DIR = ROOT_DIR / "logs"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for debug logging."""
    parser = argparse.ArgumentParser(description="Generate validation samples for Overseer review.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to logs/ with timestamped filenames.",
    )
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


def save_jsonl(path: Path, records: list[dict]) -> None:
    """Save records to JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    LOGGER.debug("Wrote %s records to %s", len(records), path)


def save_chat_template_sample(path: Path, records: list[dict]) -> None:
    """Save chat template samples to a text file."""
    with path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records, start=1):
            handle.write(f"Sample {idx}\n")
            handle.write(record["text"])
            handle.write("\n\n")
    LOGGER.debug("Wrote %s chat template samples to %s", len(records), path)


def main() -> None:
    args = parse_args()
    log_file = configure_logging(RUNTIME_LOG_DIR, args.debug, "generate_validation_sample")
    if log_file:
        LOGGER.info("Debug log file: %s", log_file)
    LOGGER.debug("Debug logging enabled")

    LOGGER.debug("Ensuring validation directory exists: %s", VALIDATION_DIR)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train.jsonl"
    val_path = PROCESSED_DIR / "val.jsonl"
    test_path = PROCESSED_DIR / "test.jsonl"

    LOGGER.debug("Train path: %s (exists=%s)", train_path, train_path.exists())
    LOGGER.debug("Val path: %s (exists=%s)", val_path, val_path.exists())
    LOGGER.debug("Test path: %s (exists=%s)", test_path, test_path.exists())
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise RuntimeError("Processed datasets not found. Run preprocess_data.py first.")

    all_records = load_jsonl(train_path) + load_jsonl(val_path) + load_jsonl(test_path)
    if not all_records:
        raise RuntimeError("No records found in processed datasets.")

    LOGGER.debug("Total records loaded: %s", len(all_records))
    LOGGER.debug("Sampling seed: %s", 42)
    random.seed(42)
    sample_50 = random.sample(all_records, k=min(50, len(all_records)))
    sample_10 = random.sample(all_records, k=min(10, len(all_records)))
    LOGGER.debug("Sample sizes: random_sample_50=%s, chat_template_sample_10=%s", len(sample_50), len(sample_10))

    save_jsonl(VALIDATION_DIR / "random_sample_50.jsonl", sample_50)
    save_chat_template_sample(VALIDATION_DIR / "chat_template_sample_10.txt", sample_10)

    LOGGER.info("Validation samples generated:")
    LOGGER.info("- %s", VALIDATION_DIR / "random_sample_50.jsonl")
    LOGGER.info("- %s", VALIDATION_DIR / "chat_template_sample_10.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Validation sample generation failed")
        raise
