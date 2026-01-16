# Phase 1 - Data Pipeline: Safe Training Data Construction

## 1. Introduction

### Learning objectives
- Explain how the dataset is normalized and randomly sampled for the 10% test run
- Run the preprocessing pipeline and interpret the console progress output
- Validate Bash syntax with Shellcheck and understand failure handling
- Apply Qwen chat templates and verify system/user/assistant role tokens
- Mask user tokens so the model learns only the assistant response
- Enforce zero-tolerance dangerous command filtering before training
- Read provenance and audit logs to verify data quality and traceability

### Plain-English explanation
This phase turns raw NL-to-Bash pairs into safe, validated training data. Think of it like airport security for commands: every record is screened before it is allowed into training.

### Why this matters
Without this step, the model could learn to generate catastrophic commands like `rm -rf /` from benign prompts. Phase 1 blocks those risks before training starts.

---

## 2. Key Concepts

### Domain terminology
- Chat template (Qwen format): The structured prompt format used by Qwen2.5-Coder.
- Zero-tolerance patterns: Regex rules that block catastrophic commands.
- Shellcheck validation: Syntax validation for Bash commands.
- Provenance tracking: The audit trail for what was filtered and why.

### Design decisions
- Dangerous pattern filtering happens before training, not just at inference, so the model never learns risky examples.
- Shellcheck catches syntax errors; safety checks catch malicious intent. Both are required.
- Every filtered record is logged for auditability and repeatability.
- Sampling is random with a fixed seed so the 10% test run is reproducible and fast to iterate.
- Full dataset runs are deferred until the pipeline is stable, saving hours during validation.

### Architecture context
This is Phase 1 (data pipeline). Training and evaluation are not in scope for this lesson.

### Data flow (7 components)
```
Load + Schema Validate
  -> Shellcheck Syntax Validation
    -> Dangerous Pattern Filtering
      -> Chat Template Application (Qwen)
        -> Tokenize + Mask (assistant-only)
          -> Deduplicate + Split (80/10/10)
            -> Save Outputs + Provenance + Validation Samples
```

---

## 3. Code Walkthrough

### File: `scripts/preprocess_data.py`

#### Imports and constants (lines 1-34)
Key constants define the dataset, tokenizer, sampling strategy, and output locations.

Lines 1-34:
```python
"""
Phase 1 data pipeline: load, validate, filter, template, tokenize, split, and record provenance.
"""
from __future__ import annotations

import json
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
```

#### Field mapping and sampling (lines 86-125)
The dataset uses `nl_command` and `bash_code`. These are normalized to `instruction` and `output`. The 10% test run is a random sample with a fixed seed.

Lines 86-125:
```python
    total_downloaded = len(records)

    # Sample dataset if SAMPLE_SIZE is set
    if SAMPLE_SIZE is not None and len(records) > SAMPLE_SIZE:
        rng = random.Random(SAMPLE_SEED)
        indices = rng.sample(range(len(records)), SAMPLE_SIZE)
        indices.sort()
        print(
            f"  Sampling {SAMPLE_SIZE} examples from {total_downloaded} total "
            f"(seed={SAMPLE_SEED})"
        )
        if hasattr(records, "select"):
            records = records.select(indices)
        else:
            records = [records[i] for i in indices]

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
```

Why normalization matters: The model training expects a consistent schema. Field mapping ensures the raw dataset can be validated and processed without errors.

#### Shellcheck validation (lines 199-270)
Shellcheck validates syntax so the model only sees correct Bash.

Lines 199-270:
```python
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
    print(f"  Validating {total} commands with shellcheck...")

    for i, example in enumerate(examples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total} ({100*(i+1)//total}%)")

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

    print(f"  Shellcheck complete: {len(valid_examples)}/{total} passed ({100*len(valid_examples)//total}%)")
    save_jsonl(LOG_DIR / "removed_invalid_syntax.jsonl", removed_examples)

    return {
        "valid_examples": valid_examples,
        "removed_examples": removed_examples,
        "shellcheck_version": shellcheck_version,
    }
```

#### Dangerous command filtering (lines 273-313)
Every command is checked against the zero-tolerance patterns before it is allowed into the dataset.

Lines 273-313:
```python
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
```

#### Chat template application (lines 328-369)
The Qwen tokenizer formats the data using its official chat template. This adds role tokens and may include a default system prompt.

Lines 328-369:
```python
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
```

#### Tokenization and masking (lines 372-437)
The pipeline masks all tokens before the assistant response so the model learns to predict only the assistant command.

Lines 372-437:
```python
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
```

#### Train/val/test split and deduplication (lines 453-497)
Duplicates are removed before the split, then a deterministic 80/10/10 split is applied.

Lines 453-497:
```python
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
        print(f"  Removed {duplicates_removed} duplicate examples")

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
```

#### Saving outputs and provenance (lines 43-47 and 505-532)
Processed datasets and provenance are saved to `data/processed/`.

Lines 43-47:
```python
def save_jsonl(path: Path, records: list[dict]) -> None:
    """Write list of dicts to JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
```

Lines 506-532:
```python
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
```

---

### File: `guardrails/patterns.py`

#### Zero-tolerance patterns (lines 8-33)
These regexes are the hard blocklist for catastrophic commands.

Lines 8-33:
```python
ZERO_TOLERANCE_PATTERNS = [
    # Root deletion
    r"rm\s+-rf\s+/",  # rm -rf /
    r"rm\s+-r\s+-f\s+/",  # rm -r -f /
    r"rm\s+--recursive\s+--force\s+/",  # rm --recursive --force /
    # Fork bomb
    r":\(\)\{:\|:\&\};:",  # fork bomb canonical
    r":\(\)\{:\|:\&\}",  # fork bomb variant
    # Disk wipe
    r"dd\s+if=/dev/zero\s+of=/dev/sd",  # zero-fill disk
    r"dd\s+if=/dev/random\s+of=/dev/sd",  # random-fill disk
    # Blanket permissions
    r"chmod\s+-R\s+777\s+/",  # chmod -R 777 /
    r"chmod\s+--recursive\s+777\s+/",  # chmod --recursive 777 /
    # Blind remote execution
    r"curl\s+.*\|\s*bash",  # curl ... | bash
    r"curl\s+.*\|\s*sh",  # curl ... | sh
    r"wget\s+.*\|\s*bash",  # wget ... | bash
    r"wget\s+.*\|\s*sh",  # wget ... | sh
    # Direct disk write
    r">\s*/dev/sd[a-z]",  # overwrite block device
    r">>\s*/dev/sd[a-z]",  # append to block device
    # Format disk
    r"mkfs\.",  # filesystem format
    r"mkswap\s+/dev/sd",  # swap format
]
```

---

## 4. Hands-On Exercises

### Prerequisites
- Python 3.8+
- Shellcheck installed (`apt-get install shellcheck` or `brew install shellcheck`)
- Dataset download works (automatic on first run)

### Exercise 1: Run the full pipeline

```bash
# Bash
cd /path/to/ready-tensor-llm
python scripts/preprocess_data.py
```

```powershell
# PowerShell
cd "C:\Projects\Ready Tensor LLM"
$env:PYTHONPATH = "."
python scripts/preprocess_data.py
```

Expected output (10% sample, seed=42):
```text
Repo card metadata block was not found. Setting CardData to empty.
Step 1/7: Loading dataset...
  Sampling 1835 examples from 18357 total (seed=42)
Step 2/7: Running shellcheck...
  Validating 1835 commands with shellcheck...
  Progress: 100/1835 (5%)
  Progress: 200/1835 (10%)
  Progress: 300/1835 (16%)
  Progress: 400/1835 (21%)
  Progress: 500/1835 (27%)
  Progress: 600/1835 (32%)
  Progress: 700/1835 (38%)
  Progress: 800/1835 (43%)
  Progress: 900/1835 (49%)
  Progress: 1000/1835 (54%)
  Progress: 1100/1835 (59%)
  Progress: 1200/1835 (65%)
  Progress: 1300/1835 (70%)
  Progress: 1400/1835 (76%)
  Progress: 1500/1835 (81%)
  Progress: 1600/1835 (87%)
  Progress: 1700/1835 (92%)
  Progress: 1800/1835 (98%)
  Shellcheck complete: 1793/1835 passed (97%)
Step 3/7: Filtering dangerous commands...
Step 4/7: Applying chat template...
Step 5/7: Tokenizing with assistant-only masking...
Step 6/7: Splitting train/val/test...
  Removed 58 duplicate examples
Step 7/7: Saving outputs...
Pipeline complete. 1388 train, 173 val, 174 test.
```

Common pitfalls:
- Shellcheck missing: the pipeline stops at Step 2 with a clear error. Install shellcheck before running.
- If you see a deprecation warning in older runs, update to the current script which uses timezone-aware timestamps.
- Field mismatch: if `nl_command` or `bash_code` are missing, those records are logged to `data/logs/missing_fields.jsonl`.
- Template confusion: Qwen may prepend a system prompt. That is expected and does not break masking.

### Exercise 2: Inspect dangerous commands log

```bash
# Bash
head -n 5 data/logs/removed_dangerous.jsonl
```

```powershell
# PowerShell
Get-Content data/logs/removed_dangerous.jsonl | Select-Object -First 5
```

Expected output:
```text
(no output; file is empty)
```

### Exercise 3: Verify train/val/test split sizes

```bash
# Bash
wc -l data/processed/*.jsonl
```

```powershell
# PowerShell
Get-ChildItem data/processed/*.jsonl | ForEach-Object { "$($_.Name): $(Get-Content $_.FullName | Measure-Object -Line | Select-Object -ExpandProperty Lines)" }
```

Expected output:
```text
train.jsonl: 1388
val.jsonl: 173
test.jsonl: 174
```

### Exercise 4: Inspect provenance (audit trail)

```bash
# Bash
cat data/processed/provenance.json
```

```powershell
# PowerShell
Get-Content data/processed/provenance.json
```

Expected output:
```json
{
  "source_dataset": "prabhanshubhowal/natural_language_to_linux",
  "source_url": "https://huggingface.co/datasets/prabhanshubhowal/natural_language_to_linux",
  "collection_date": "2026-01-16T00:49:04.887790+00:00",
  "total_examples_downloaded": 18357,
  "schema_violations": 0,
  "missing_fields": 0,
  "invalid_syntax": 42,
  "dangerous_commands": 0,
  "final_train_size": 1388,
  "final_val_size": 173,
  "final_test_size": 174,
  "shellcheck_version": "ShellCheck - shell script analysis tool\r\nversion: 0.9.0\r\nlicense: GNU General Public License, version 3\r\nwebsite: https://www.shellcheck.net",
  "shellcheck_pass_rate": 97.71,
  "data_hash_sha256": "cf93171dfb26ab6f193b9493dfd28e6b332beec99a5a67f501623b0edc4b6a20",
  "filtering_config": {
    "zero_tolerance_patterns": [
      "rm\\s+-rf\\s+/",
      "rm\\s+-r\\s+-f\\s+/",
      "rm\\s+--recursive\\s+--force\\s+/",
      ":\\(\\)\\{:\\|:\\&\\};:",
      ":\\(\\)\\{:\\|:\\&\\}",
      "dd\\s+if=/dev/zero\\s+of=/dev/sd",
      "dd\\s+if=/dev/random\\s+of=/dev/sd",
      "chmod\\s+-R\\s+777\\s+/",
      "chmod\\s+--recursive\\s+777\\s+/",
      "curl\\s+.*\\|\\s*bash",
      "curl\\s+.*\\|\\s*sh",
      "wget\\s+.*\\|\\s*bash",
      "wget\\s+.*\\|\\s*sh",
      ">\\s*/dev/sd[a-z]",
      ">>\\s*/dev/sd[a-z]",
      "mkfs\\.",
      "mkswap\\s+/dev/sd"
    ],
    "chat_template_source": "Qwen/Qwen2.5-Coder-7B-Instruct tokenizer.chat_template",
    "masking_strategy": "assistant_only (user tokens = -100)",
    "sample_size": 1835,
    "sample_seed": 42
  }
}
```

### Exercise 5: Inspect a tokenized example

```bash
# Bash
python -c "import json; print(json.loads(open('data/processed/train.jsonl', 'r').readline())['text'])"
```

```powershell
# PowerShell
python -c "import json; print(json.loads(open('data/processed/train.jsonl', 'r').readline())['text'])"
```

Expected output:
```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
List all files/directories under current directory using comma  as the delimiter for different fields in the output<|im_end|>
<|im_start|>assistant
find . -ls|awk 'BEGIN{OFS=","}$1=$1'<|im_end|>
```

---

## 5. Interview Preparation

### Question 1
"Walk me through how you ensure training data does not contain dangerous commands."

Model answer:
"I use a zero-tolerance filtering step before training. The patterns live in `guardrails/patterns.py` and cover catastrophic actions like root deletion, disk wipes, and remote execution. In `scripts/preprocess_data.py`, every command is checked with `is_dangerous_command()` and any matches are logged to `data/logs/removed_dangerous.jsonl` and removed. I then run a final verification pass to ensure no dangerous commands remain. This means the model never sees those commands during training, which is safer than only filtering at inference time."

### Question 2
"Why do you validate with Shellcheck if you are already filtering dangerous patterns?"

Model answer:
"Shellcheck and dangerous pattern filtering solve different problems. Dangerous patterns catch harmful intent. Shellcheck catches invalid Bash syntax. A command might be safe but syntactically broken, and I do not want the model to learn invalid syntax. By running both checks, the dataset contains commands that are both safe and valid."

### Question 3
"How do you handle the tradeoff between data quality and dataset size after filtering?"

Model answer:
"I track the filtering statistics and preserve them in provenance. For the 10% sample run, 42 commands failed Shellcheck, 58 duplicates were removed, and 0 dangerous commands were found. The final dataset still has 1,735 unique examples, which is a good tradeoff. If a future dataset lost too many examples, the logs show exactly why and I could adjust patterns or cleaning rules, but safety filtering remains non-negotiable."

### Question 4
"Explain your chat template and why it matters for instruction fine-tuning."

Model answer:
"I use Qwen's official chat template via the tokenizer. The formatted text includes `<|im_start|>` and `<|im_end|>` tokens and typically includes a system prompt, plus user and assistant roles. During tokenization I mask everything before the assistant response, so the model only learns to generate the command, not to complete the user instruction. That keeps the fine-tuned behavior aligned with the base model's chat format."

### Question 5
"How would you extend this pipeline to support Docker or Git commands?"

Model answer:
"I would add shell-type detection during normalization and expand the dangerous patterns with Docker and Git-specific risks. Shellcheck is Bash-only, so I would integrate linters like hadolint for Docker and git command validation rules. Each command type could be stored as metadata and potentially trained as separate adapters if distributions diverge."

### Question 6
"Why is deduplication done before the train/val/test split?"

Model answer:
"Deduplication happens before the split to prevent leakage across splits. If duplicates are removed after splitting, identical examples could land in both train and validation, inflating metrics. Removing duplicates first guarantees split integrity and makes leakage checks deterministic."

---

## 6. Key Takeaways

- Phase 1 builds safe training data through validation and filtering
- Zero-tolerance patterns block catastrophic commands before training
- Shellcheck validates syntax correctness (safety is not the same as correctness)
- Qwen chat templates preserve the base model's instruction format
- Masking ensures the model learns to generate commands, not user prompts
- Provenance and logs provide a full audit trail
- Sampling is randomized and reproducible via seed

---

## 7. Summary Reference Card

### Input
- HuggingFace dataset: `prabhanshubhowal/natural_language_to_linux`
- Raw fields: `nl_command`, `bash_code`, optional `input`

### Output
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/provenance.json`
- `data/logs/schema_violations.jsonl`
- `data/logs/missing_fields.jsonl`
- `data/logs/removed_invalid_syntax.jsonl`
- `data/logs/removed_dangerous.jsonl`
- `data/validation/random_sample_50.jsonl`
- `data/validation/chat_template_sample_10.txt`

### Key functions
- `load_and_validate_dataset()` - mapping and sampling
- `validate_bash_syntax()` - Shellcheck validation
- `filter_invalid_syntax()` - remove invalid commands
- `filter_dangerous_commands()` - remove dangerous commands
- `apply_chat_template()` - Qwen chat formatting
- `tokenize_with_masking()` - mask user tokens
- `split_dataset()` - deduplicate and split
- `save_jsonl()` - write JSONL outputs
- `record_provenance()` - audit trail

### Error handling
- Missing fields -> `data/logs/missing_fields.jsonl`
- Schema violations -> `data/logs/schema_violations.jsonl`
- Shellcheck failures -> `data/logs/removed_invalid_syntax.jsonl`
- Dangerous commands -> `data/logs/removed_dangerous.jsonl`
- Download failures -> pipeline aborts after retries

### Configuration
- `MAX_LENGTH = 2048`
- `SAMPLE_SIZE = 1835`
- `SAMPLE_SEED = 42`
- `SPLIT_SEED = 42`
- `SHELLCHECK_TIMEOUT = 5`
- Change `SAMPLE_SEED` to mix the 10% sample between runs.

### Dependencies
- `datasets` (HuggingFace datasets)
- `transformers` (Qwen tokenizer)
- `pydantic` (schema validation)
- `shellcheck` (Bash syntax validation)

---

## 8. Next Steps

- Review `data/logs/` to understand filtered records.
- Review `data/validation/` artifacts for Overseer spot checks.
- Generate validation samples with `python scripts/generate_validation_sample.py`.
- Keep the 10% sample run until full dataset processing is required.

---

## 9. General Best Practices (Not All Implemented Here)

### Schema and policy checks
- Implemented: Pydantic schema validation
- Implemented: Zero-tolerance dangerous pattern filtering
- Not implemented: Inference-time policy enforcement

### Prompt sanitization
- Implemented: Training data sanitization via filtering
- Not implemented: Input sanitization at inference time
- Not implemented: Quarantine mode for suspicious inputs

### Semantic guardrails
- Not implemented: Risk scoring (safe/review/dangerous)
- Not implemented: Semantic checks beyond regex patterns
- Not implemented: Confidence thresholding for output

### Provenance and observability
- Implemented: Provenance file for the final dataset
- Implemented: Audit logs for removed records
- Not implemented: External dataset version tracking (WandB or HF dataset versioning)

Note: Phase 1 focuses on training data safety. Inference-time guardrails and risk scoring are not part of the current implementation.
