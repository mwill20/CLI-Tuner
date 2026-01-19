"""
Shared helpers for Phase 3 evaluation scripts.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    """Save records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def save_json(path: Path, payload: dict) -> None:
    """Save JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def normalize_command(text: str) -> str:
    """Normalize command text for exact-match evaluation."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(predicted: str, reference: str) -> bool:
    """Return True if normalized predicted matches reference."""
    return normalize_command(predicted) == normalize_command(reference)


def is_command_only(text: str) -> bool:
    """Heuristic check for command-only outputs (no prose)."""
    if not text or not text.strip():
        return False
    stripped = text.strip()
    lower = stripped.lower()
    if "```" in stripped:
        return False
    explanation_markers = [
        "here's",
        "here is",
        "this command",
        "you can",
        "to do this",
        "the command",
        "try this",
        "use this",
        "this will",
        "explanation",
        "explain",
        "steps",
        "note:",
        "command:",
        "here ",
        "use ",
    ]
    if any(marker in lower for marker in explanation_markers):
        return False

    if stripped.endswith((".", "!", "?")):
        return False

    lines = [line for line in stripped.split("\n") if line.strip()]
    if len(lines) > 5:
        return False
    if len(lines) > 1:
        has_continuation = any(line.rstrip().endswith("\\") for line in lines[:-1])
        has_pipe = any("|" in line for line in lines)
        if not has_continuation and not has_pipe:
            return False
    return True


def shellcheck_available() -> bool:
    """Return True if shellcheck is in PATH."""
    return shutil.which("shellcheck") is not None


def validate_bash_syntax(command: str) -> Optional[bool]:
    """Validate syntax with shellcheck (returns None if unavailable)."""
    if not shellcheck_available():
        return None
    try:
        result = subprocess.run(
            ["shellcheck", "--shell=bash", "--severity=error", "--format=json", "-"],
            input=command.encode("utf-8"),
            capture_output=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    if result.returncode == 0:
        return True
    return False


def syntax_validity_rate(
    commands: list[str],
    validator: Callable[[str], Optional[bool]] = validate_bash_syntax,
) -> Optional[float]:
    """Compute syntax validity rate (None if no validations)."""
    results: list[bool] = []
    for command in commands:
        verdict = validator(command)
        if verdict is None:
            continue
        results.append(verdict)
    if not results:
        return None
    return round(sum(1 for r in results if r) / len(results), 4)


def command_only_rate(outputs: list[str]) -> float:
    """Compute command-only rate for a list of outputs."""
    if not outputs:
        return 0.0
    count = sum(1 for output in outputs if is_command_only(output))
    return round(count / len(outputs), 4)


def exact_match_rate(predictions: list[str], references: list[str]) -> float:
    """Compute exact-match rate for predictions vs references."""
    if not predictions:
        return 0.0
    matches = sum(
        1 for predicted, reference in zip(predictions, references)
        if exact_match(predicted, reference)
    )
    return round(matches / len(predictions), 4)


def compute_domain_metrics(
    predictions: list[str],
    references: list[str],
    syntax_validator: Callable[[str], Optional[bool]] = validate_bash_syntax,
) -> dict:
    """Compute domain metrics for evaluation."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    exact_matches = sum(
        1 for predicted, reference in zip(predictions, references)
        if exact_match(predicted, reference)
    )
    command_count = sum(1 for output in predictions if is_command_only(output))

    exact_rate = round(exact_matches / len(predictions), 4) if predictions else 0.0
    command_rate = round(command_count / len(predictions), 4) if predictions else 0.0
    syntax_rate = syntax_validity_rate(predictions, validator=syntax_validator)

    return {
        "total": len(predictions),
        "exact_match": exact_rate,
        "command_only_rate": command_rate,
        "syntax_validity": syntax_rate,
        "exact_match_count": exact_matches,
        "command_only_count": command_count,
    }
