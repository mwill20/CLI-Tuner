# Phase 1: Data Pipeline Specification
**Version:** 1.0  
**Created:** 2026-01-15  
**Status:** Ready for Overseer Review  
**Addresses:** OSV-001 (Trust Boundary Validation), OSV-003 (Data Preprocessing Verification), OSV-007 (Phase 1 Handoff)  
**Dependencies:** Phase 0 complete  
**Duration:** 1-2 weeks  
**Owner:** Primary Engineer

---

## PURPOSE

Build secure, reproducible data pipeline that:
1. Downloads dataset from HuggingFace Hub
2. Validates syntax (shellcheck)
3. Filters dangerous commands (zero-tolerance)
4. Applies chat template (Qwen2.5 format)
5. Verifies assistant-only masking
6. Splits train/val/test
7. Records full provenance

**Output:** Clean, validated dataset ready for training (Phase 2).

---

## ARCHITECTURE OVERVIEW

```
Data Pipeline Flow:

HuggingFace Hub                           TRUST BOUNDARY #1
    ↓                                    (Schema Validation)
[1. Dataset Loading]
    ↓
Raw Dataset (JSON records)                TRUST BOUNDARY #2
    ↓                                    (Syntax Validation)
[2. Shellcheck Validation]
    ↓
Syntax-Valid Commands                     TRUST BOUNDARY #3
    ↓                                    (Dangerous Pattern Filtering)
[3. Dangerous Command Filtering]
    ↓
Safe Commands Only                        TRUST BOUNDARY #4
    ↓                                    (Chat Template Application)
[4. Chat Template Formatting]
    ↓
Qwen2.5 Formatted Examples                TRUST BOUNDARY #5
    ↓                                    (Masking Verification)
[5. Assistant-Only Masking]
    ↓
Tokenized with Labels                     TRUST BOUNDARY #6
    ↓                                    (Split Validation)
[6. Train/Val/Test Split]
    ↓
Final Datasets + Provenance
```

---

## COMPONENT 1: DATASET LOADING (OSV-001 Partial)

### Data Source
```yaml
source:
  provider: HuggingFace Hub
  dataset_id: prabhanshubhowal/natural_language_to_linux
  subset: null  # Use default split
  cache_dir: ~/.cache/huggingface/datasets  # HF default
  trust_remote_code: false  # Security: no custom code execution
```

### Schema Validation
```python
# Expected schema
{
  "instruction": str,  # Natural language query
  "input": str,        # Additional context (often empty)
  "output": str        # Bash command (reference answer)
}

# Validation rules
validation_rules:
  instruction:
    type: str
    min_length: 3
    max_length: 500
    nullable: false
  input:
    type: str
    max_length: 200
    nullable: true  # Can be empty string
  output:
    type: str
    min_length: 1
    max_length: 500
    nullable: false
```

### Field Normalization & Sampling
```yaml
field_mapping:
  nl_command: instruction
  bash_code: output
  input: input  # optional, defaults to empty string

sampling:
  sample_size: 1835  # 10% test run (configurable)
  sample_seed: 42
  strategy: random sample from full dataset
  full_run: set sample_size = null
  mix_sample: change sample_seed and rerun
```

### Error Handling (OSV-001)
```yaml
on_schema_violation:
  action: remove_record
  logging:
    file: data/logs/schema_violations.jsonl
    fields: [index, instruction, input, output, violation_reason]
  
on_missing_field:
  action: remove_record
  logging:
    file: data/logs/missing_fields.jsonl
    fields: [index, missing_field]

on_download_failure:
  action: abort_pipeline
  retry: 3
  backoff: exponential
  error_message: "Failed to download dataset after 3 retries"
```

### Implementation
```python
from datasets import load_dataset
from pydantic import BaseModel, Field, ValidationError
import json
import random

class BashCommandExample(BaseModel):
    """Schema for raw dataset examples."""
    instruction: str = Field(min_length=3, max_length=500)
    input: str = Field(default="", max_length=200)
    output: str = Field(min_length=1, max_length=500)

def load_and_validate_dataset(dataset_id: str) -> dict:
    """
    Load dataset from HuggingFace Hub and validate schema.
    
    Returns:
        dict with keys: valid_examples, schema_violations, missing_fields
    """
    dataset = load_dataset(dataset_id, trust_remote_code=False)
    
    records = dataset["train"]
    total_downloaded = len(records)

    # Sample dataset if SAMPLE_SIZE is set
    if SAMPLE_SIZE is not None and len(records) > SAMPLE_SIZE:
        rng = random.Random(SAMPLE_SEED)
        indices = rng.sample(range(len(records)), SAMPLE_SIZE)
        indices.sort()
        records = records.select(indices)

    valid_examples = []
    schema_violations = []
    missing_fields = []

    field_mapping = {
        "nl_command": "instruction",
        "bash_code": "output",
    }
    
    for idx, record in enumerate(records):
        normalized_record = {}
        for old_field, new_field in field_mapping.items():
            if old_field in record:
                normalized_record[new_field] = record[old_field]
        normalized_record["input"] = record.get("input", "")
        try:
            validated = BashCommandExample(**normalized_record)
            valid_examples.append(validated.dict())
        except ValidationError as e:
            schema_violations.append({
                "index": idx,
                "record": normalized_record,
                "violation": str(e)
            })
        except KeyError as e:
            missing_fields.append({
                "index": idx,
                "missing_field": str(e)
            })
    
    # Log violations
    with open("data/logs/schema_violations.jsonl", "w") as f:
        for violation in schema_violations:
            f.write(json.dumps(violation) + "\n")
    
    with open("data/logs/missing_fields.jsonl", "w") as f:
        for missing in missing_fields:
            f.write(json.dumps(missing) + "\n")
    
    return {
        "valid_examples": valid_examples,
        "schema_violations": schema_violations,
        "missing_fields": missing_fields
    }
```

### Acceptance Criteria
- [ ] Dataset downloads successfully from HuggingFace Hub
- [ ] Schema validation using Pydantic (BashCommandExample model)
- [ ] Field mapping applied (nl_command -> instruction, bash_code -> output)
- [ ] All violations logged to `data/logs/schema_violations.jsonl`
- [ ] Missing fields logged to `data/logs/missing_fields.jsonl`
- [ ] Pipeline aborts if download fails after 3 retries
- [ ] If sampling enabled, selection is randomized with seed for reproducibility
- [ ] Valid examples count ≥ 500 (minimum threshold for training)
- [ ] If post-filtering dataset < 500 examples, pipeline logs warning and continues (training may be suboptimal but not blocked)

---

## COMPONENT 2: SHELLCHECK VALIDATION (OSV-003 Partial)

### Shellcheck Configuration
```yaml
shellcheck:
  version: ">=0.9.0"
  shell: bash
  severity: error  # Only block on errors, not warnings
  exclude_codes: []  # No exclusions (strict)
  timeout_per_command: 5  # seconds
```

### Validation Strategy
```yaml
validation:
  tool: shellcheck
  method: subprocess call per command
  input_format: stdin (pipe command to shellcheck)
  output_format: json
  
  pass_criteria:
    exit_code: 0
    no_errors: true
  
  fail_action:
    remove_from_dataset: true
    log_to_file: data/logs/removed_invalid_syntax.jsonl
    log_fields: [instruction, output, shellcheck_errors]
```

### Implementation
```python
import subprocess
import json

def validate_bash_syntax(command: str) -> dict:
    """
    Validate Bash command syntax using shellcheck.
    
    Returns:
        dict with keys: valid (bool), errors (list)
    """
    try:
        result = subprocess.run(
            ["shellcheck", "--shell=bash", "--severity=error", "--format=json", "-"],
            input=command.encode(),
            capture_output=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return {"valid": True, "errors": []}
        else:
            errors = json.loads(result.stdout.decode())
            return {"valid": False, "errors": errors}
    
    except subprocess.TimeoutExpired:
        return {"valid": False, "errors": [{"message": "Shellcheck timeout"}]}
    except FileNotFoundError:
        raise RuntimeError("Shellcheck not installed. Install: apt-get install shellcheck")

def filter_invalid_syntax(examples: list) -> dict:
    """
    Filter examples with invalid syntax.
    
    Returns:
        dict with keys: valid_examples, removed_examples
    """
    valid_examples = []
    removed_examples = []
    
    for example in examples:
        command = example["output"]
        validation = validate_bash_syntax(command)
        
        if validation["valid"]:
            valid_examples.append(example)
        else:
            removed_examples.append({
                "instruction": example["instruction"],
                "output": command,
                "shellcheck_errors": validation["errors"]
            })
    
    # Log removed examples
    with open("data/logs/removed_invalid_syntax.jsonl", "w") as f:
        for removed in removed_examples:
            f.write(json.dumps(removed) + "\n")
    
    return {
        "valid_examples": valid_examples,
        "removed_examples": removed_examples
    }
```

### Pre-Flight Check
```python
def check_shellcheck_installed() -> str:
    """Verify shellcheck is installed and get version."""
    try:
        result = subprocess.run(
            ["shellcheck", "--version"],
            capture_output=True,
            timeout=5
        )
        version_output = result.stdout.decode()
        # Extract version (e.g., "version: 0.9.0")
        return version_output
    except FileNotFoundError:
        raise RuntimeError(
            "Shellcheck not installed. Install: "
            "apt-get install shellcheck (Linux) or brew install shellcheck (Mac)"
        )
```

### Acceptance Criteria
- [ ] Pre-flight check runs before processing (check_shellcheck_installed)
- [ ] Shellcheck version ≥ 0.9.0 verified
- [ ] All commands validated (exit code 0 = pass)
- [ ] Syntax errors logged to `data/logs/removed_invalid_syntax.jsonl`
- [ ] Shellcheck timeout set to 5 seconds per command
- [ ] Pipeline aborts with clear error message if shellcheck not installed
- [ ] Shellcheck pass rate ≥ 95% (on final dataset)

---

## COMPONENT 3: DANGEROUS COMMAND FILTERING (OSV-003 Critical)

### Zero-Tolerance Patterns (OSV-003)
```python
ZERO_TOLERANCE_PATTERNS = [
    # Root deletion
    r'rm\s+-rf\s+/',
    r'rm\s+-r\s+-f\s+/',
    r'rm\s+--recursive\s+--force\s+/',
    
    # Fork bomb
    r':\(\)\{:\|:\&\};:',
    r':\(\)\{:\|:\&\}',
    
    # Disk wipe
    r'dd\s+if=/dev/zero\s+of=/dev/sd',
    r'dd\s+if=/dev/random\s+of=/dev/sd',
    
    # Blanket permissions
    r'chmod\s+-R\s+777\s+/',
    r'chmod\s+--recursive\s+777\s+/',
    
    # Blind remote execution
    r'curl\s+.*\|\s*bash',
    r'curl\s+.*\|\s*sh',
    r'wget\s+.*\|\s*bash',
    r'wget\s+.*\|\s*sh',
    
    # Direct disk write
    r'>\s*/dev/sd[a-z]',
    r'>>\s*/dev/sd[a-z]',
    
    # Format disk
    r'mkfs\.',
    r'mkswap\s+/dev/sd',
]
```

### Pattern Matching Strategy
```yaml
pattern_matching:
  engine: regex (Python re module)
  flags: re.IGNORECASE
  compilation: pre-compile all patterns (performance)
  execution_order: sequential (stop on first match)
  
  on_match:
    action: remove_from_dataset
    log_to_file: data/logs/removed_dangerous.jsonl
validation/
    random_sample_50.jsonl            # Dangerous pattern verification sample
    chat_template_sample_10.txt       # Manual chat template review
    log_fields: [instruction, output, pattern_matched]
  
  verification:
    final_dataset_check: true
    random_sample_size: 50
    manual_review: recommended (Overseer spot-check)
```

### Implementation
```python
import re

# Pre-compile patterns (performance)
COMPILED_PATTERNS = [(pattern, re.compile(pattern, re.IGNORECASE)) for pattern in ZERO_TOLERANCE_PATTERNS]

def is_dangerous_command(command: str) -> dict:
    """
    Check if command matches any zero-tolerance pattern.
    
    Returns:
        dict with keys: is_dangerous (bool), pattern_matched (str or None)
    """
    for pattern_str, pattern_re in COMPILED_PATTERNS:
        if pattern_re.search(command):
            return {"is_dangerous": True, "pattern_matched": pattern_str}
    
    return {"is_dangerous": False, "pattern_matched": None}

def filter_dangerous_commands(examples: list) -> dict:
    """
    Filter examples with dangerous commands.
    
    Returns:
        dict with keys: safe_examples, removed_examples
    """
    safe_examples = []
    removed_examples = []
    
    for example in examples:
        command = example["output"]
        check = is_dangerous_command(command)
        
        if not check["is_dangerous"]:
            safe_examples.append(example)
        else:
            removed_examples.append({
                "instruction": example["instruction"],
                "output": command,
                "pattern_matched": check["pattern_matched"]
            })
    
    # Log removed examples
    with open("data/logs/removed_dangerous.jsonl", "w") as f:
        for removed in removed_examples:
            f.write(json.dumps(removed) + "\n")
    
    # CRITICAL: Verify no dangerous commands remain
    for example in safe_examples:
        check = is_dangerous_command(example["output"])
        if check["is_dangerous"]:
            raise RuntimeError(f"Dangerous command slipped through: {example['output']}")
    
    return {
        "safe_examples": safe_examples,
        "removed_examples": removed_examples
    }
```

### Acceptance Criteria (OSV-003)
- [ ] All zero-tolerance patterns defined and pre-compiled
- [ ] Pattern matching uses `re.IGNORECASE` flag
- [ ] Dangerous commands logged to `data/logs/removed_dangerous.jsonl`
- [ ] **CRITICAL:** Final dataset verification (no dangerous commands remain)
- [ ] Verification uses random sample (n=50) for Overseer spot-check
- [ ] Pipeline aborts if dangerous command found in final dataset

---

## COMPONENT 4: CHAT TEMPLATE APPLICATION (OSV-003)

### Qwen2.5 Chat Template
```yaml
chat_template:
  format: Qwen2.5 official template
  source: tokenizer.chat_template (from base model)
  structure: "<|im_start|>system\n<default_system_prompt><|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
  
  verification:
    method: compare against tokenizer.chat_template
    assert_tokens_present: ["<|im_start|>", "<|im_end|>", "user", "assistant"]
```

### Template Application
```python
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

def apply_chat_template(examples: list) -> list:
    """
    Apply Qwen2.5 chat template to all examples.
    
    Returns:
        list of examples with 'text' field (formatted prompt)
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    
    # Verify chat template exists
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise RuntimeError("Tokenizer does not have chat_template attribute")
    
    formatted_examples = []
    
    for example in examples:
        # Construct chat format
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        
        # Apply template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Verify format (Qwen uses <|im_start|> and <|im_end|>)
        assert "<|im_start|>" in formatted_text, "im_start token missing"
        assert "<|im_end|>" in formatted_text, "im_end token missing"
        assert "user" in formatted_text and "assistant" in formatted_text, "Role tokens missing"
        
        formatted_examples.append({
            "instruction": example["instruction"],
            "output": example["output"],
            "text": formatted_text
        })
    
    return formatted_examples
```

### Verification Strategy (OSV-003)
```yaml
verification:
  sample_size: 50  # Random sample for manual review
  checks:
    - chat_template_correctness: all examples have im_start/im_end and role tokens
    - no_template_leakage: no examples have raw {instruction} or {output} placeholders
    - consistent_formatting: all examples follow same structure
    - system_prompt_allowed: tokenizer may prepend default system message
  
  manual_review:
    reviewer: Overseer
    method: spot-check 10 random examples
    criteria: visual inspection of formatted text
```

### Acceptance Criteria
- [ ] Tokenizer loaded from `Qwen/Qwen2.5-Coder-7B-Instruct`
- [ ] Chat template applied to all examples
- [ ] All examples contain `<|im_start|>` and `<|im_end|>` tokens with user/assistant roles
- [ ] No raw placeholders (`{instruction}`, `{output}`) in formatted text
- [ ] Random sample (n=50) logged for Overseer verification
- [ ] Overseer spot-checks 10 examples for correctness

---

## COMPONENT 5: ASSISTANT-ONLY MASKING (OSV-003)

### Masking Strategy
```yaml
masking:
  objective: train model to predict assistant responses only
  method: set user tokens to -100 in labels
  
  tokenization:
    tokenizer: Qwen/Qwen2.5-Coder-7B-Instruct
    max_length: 2048
    padding: false  # Will be done during training
    truncation: true
  
  label_construction:
    user_tokens: -100  # Ignored in loss calculation
    assistant_tokens: actual token IDs
    im_end_tokens: actual token IDs (kept)
```

### Implementation
```python
def tokenize_with_masking(examples: list) -> list:
    """
    Tokenize examples and apply assistant-only masking.
    
    Returns:
        list of examples with 'input_ids' and 'labels'
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)

    tokenized_examples = []

    for example in examples:
        # Tokenize full text
        full_encoding = tokenizer(
            example["text"],
            max_length=2048,
            truncation=True,
            return_tensors=None  # Return lists, not tensors
        )

        input_ids = full_encoding["input_ids"]

        # Create labels (copy of input_ids)
        labels = input_ids.copy()

        # Find assistant marker in text
        assistant_marker = "<|im_start|>assistant\\n"
        assistant_pos = example["text"].find(assistant_marker)
        if assistant_pos == -1:
            raise RuntimeError(f"Assistant marker not found in: {example['text']}")

        # Tokenize up to assistant marker to find split point
        pre_assistant = example["text"][:assistant_pos + len(assistant_marker)]
        pre_tokens = tokenizer(pre_assistant, return_tensors=None)["input_ids"]
        assistant_start_idx = len(pre_tokens)

        # Mask user tokens (everything before assistant)
        labels[:assistant_start_idx] = [-100] * assistant_start_idx

        tokenized_examples.append({
            "instruction": example["instruction"],
            "output": example["output"],
            "text": example["text"],
            "input_ids": input_ids,
            "labels": labels
        })

    return tokenized_examples
```

### Verification (OSV-003)
```python
def verify_masking(tokenized_examples: list) -> bool:
    """
    Verify assistant-only masking is correct.
    
    Returns:
        bool (True if all examples pass verification)
    """
    for example in tokenized_examples:
        labels = example["labels"]

        masked_count = sum(1 for label in labels if label == -100)
        unmasked_count = sum(1 for label in labels if label != -100)

        assert masked_count > 0, "User tokens not masked"
        assert unmasked_count > 0, "Assistant tokens should not be masked"

    return True
```

### Acceptance Criteria
- [ ] All examples tokenized with max_length=2048
- [ ] User tokens (before `<|im_start|>assistant`) set to -100 in labels
- [ ] Assistant tokens (after `<|im_start|>assistant`) retain actual token IDs
- [ ] Verification function passes for all examples
- [ ] Masking verified programmatically (not manual inspection)

---

## COMPONENT 6: TRAIN/VAL/TEST SPLIT (OSV-007)

### Split Configuration
```yaml
split:
  method: random_split (sklearn or datasets.train_test_split)
  seed: 42  # Reproducibility
  ratios:
    train: 0.8
    val: 0.1
    test: 0.1
  
  stratification: none  # Dataset likely too small for stratification
  
  leakage_check:
    method: verify no overlap between splits using full-record hash
    record_key: sha256(instruction + input + output)
    assertion: set(train_fingerprints) & set(val_fingerprints) == set()
    assertion: set(train_fingerprints) & set(test_fingerprints) == set()
    assertion: set(val_fingerprints) & set(test_fingerprints) == set()

  deduplication:
    enabled: true
    record_key: sha256(instruction + input + output)
    action: remove duplicates before split
```

### Implementation
```python
from datasets import Dataset
import hashlib
import json
import random

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

def split_dataset(examples: list, seed: int = 42) -> dict:
    """
    Split dataset into train/val/test.
    
    Returns:
        dict with keys: train, val, test (each is a list of examples)
    """
    # Deduplicate using full-record fingerprint
    seen_fingerprints = set()
    unique_examples = []

    for example in examples:
        fp = record_fingerprint(example)
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            unique_examples.append(example)

    examples = unique_examples

    random.seed(seed)
    random.shuffle(examples)
    
    n = len(examples)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train = examples[:train_size]
    val = examples[train_size:train_size + val_size]
    test = examples[train_size + val_size:]
    
    # Leakage check (using full-record fingerprint)
    train_fingerprints = {record_fingerprint(ex) for ex in train}
    val_fingerprints = {record_fingerprint(ex) for ex in val}
    test_fingerprints = {record_fingerprint(ex) for ex in test}
    
    assert len(train_fingerprints & val_fingerprints) == 0, "Train-Val leakage detected"
    assert len(train_fingerprints & test_fingerprints) == 0, "Train-Test leakage detected"
    assert len(val_fingerprints & test_fingerprints) == 0, "Val-Test leakage detected"
    
    return {
        "train": train,
        "val": val,
        "test": test
    }
```

### Acceptance Criteria
- [ ] Random split with seed=42 (reproducibility)
- [ ] Split ratios: 80% train, 10% val, 10% test
- [ ] No data leakage between splits (verified via full-record hash)
- [ ] Duplicates removed before split (logged count)
- [ ] Minimum dataset size: 500 examples (ensures train ≥ 400, val ≥ 50, test ≥ 50)

---

## COMPONENT 7: PROVENANCE RECORDING (OSV-003)

### Provenance Schema
```yaml
provenance:
  file: data/processed/provenance.json
  format: JSON
  
  fields:
    source_dataset: str  # HuggingFace dataset ID
    source_url: str      # Full URL
    collection_date: ISO8601 datetime
    
    total_examples_downloaded: int
    schema_violations: int
    missing_fields: int
    invalid_syntax: int
    dangerous_commands: int
    
    final_train_size: int
    final_val_size: int
    final_test_size: int
    
    shellcheck_version: str
    shellcheck_pass_rate: float  # percentage
    
    data_hash_sha256: str  # Hash of final train.jsonl
    
    filtering_config:
      zero_tolerance_patterns: list[str]
      chat_template_source: str
      masking_strategy: str
```

### Implementation
```python
import hashlib
from datetime import datetime, timezone
import json

def compute_dataset_hash(examples: list) -> str:
    """Compute SHA256 hash of dataset for provenance."""
    data_str = json.dumps(examples, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

def record_provenance(stats: dict, final_splits: dict) -> None:
    """
    Record full data provenance.
    
    Args:
        stats: dict with processing statistics
        final_splits: dict with train/val/test splits
    """
    provenance = {
        "source_dataset": "prabhanshubhowal/natural_language_to_linux",
        "source_url": "https://huggingface.co/datasets/prabhanshubhowal/natural_language_to_linux",
        "collection_date": datetime.now(timezone.utc).isoformat(),
        
        "total_examples_downloaded": stats["total_downloaded"],
        "schema_violations": stats["schema_violations"],
        "missing_fields": stats["missing_fields"],
        "invalid_syntax": stats["invalid_syntax"],
        "dangerous_commands": stats["dangerous_commands"],
        
        "final_train_size": len(final_splits["train"]),
        "final_val_size": len(final_splits["val"]),
        "final_test_size": len(final_splits["test"]),
        
        "shellcheck_version": stats["shellcheck_version"],
        "shellcheck_pass_rate": stats["shellcheck_pass_rate"],
        
        "data_hash_sha256": compute_dataset_hash(final_splits["train"]),
        
        "filtering_config": {
            "zero_tolerance_patterns": ZERO_TOLERANCE_PATTERNS,
            "chat_template_source": "Qwen/Qwen2.5-Coder-7B-Instruct tokenizer.chat_template",
            "masking_strategy": "assistant_only (user tokens = -100)",
            "sample_size": SAMPLE_SIZE,
            "sample_seed": SAMPLE_SEED
        }
    }
    
    with open("data/processed/provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
```

### Acceptance Criteria
- [ ] Provenance file created at `data/processed/provenance.json`
- [ ] All required fields present (source, dates, counts, hashes)
- [ ] Data hash computed using SHA256
- [ ] Shellcheck version recorded
- [ ] Zero-tolerance patterns recorded in filtering_config

---

## COMPONENT 8: VALIDATION SAMPLES (OVERSEER)

Generate validation samples for Overseer review.

### Script
```bash
python scripts/generate_validation_sample.py
```

### Outputs
- `data/validation/random_sample_50.jsonl` (dangerous pattern verification)
- `data/validation/chat_template_sample_10.txt` (manual chat template review)

---

## FINAL OUTPUTS

### File Structure
```
data/
  raw/                                   # (Empty - HF cache used)
  processed/
    train.jsonl                          # Tokenized examples (input_ids, labels)
    val.jsonl
    test.jsonl
    provenance.json                      # Full audit trail
  logs/
    schema_violations.jsonl              # Schema validation failures
    missing_fields.jsonl                 # Missing field errors
    removed_invalid_syntax.jsonl         # Shellcheck failures
    removed_dangerous.jsonl              # Dangerous command matches
  validation/
    random_sample_50.jsonl               # Dangerous pattern verification sample
    chat_template_sample_10.txt          # Manual chat template review
```

### Dataset Format (train.jsonl, val.jsonl, test.jsonl)
```json
{
  "instruction": "List all PDF files in current directory",
  "output": "find . -maxdepth 1 -name '*.pdf'",
  "text": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nList all PDF files in current directory<|im_end|>\n<|im_start|>assistant\nfind . -maxdepth 1 -name '*.pdf'<|im_end|>",
  "input_ids": [101, 2023, ...],
  "labels": [-100, -100, ..., 2023, ...]
}
```

---

## OVERSEER VALIDATION CHECKLIST (OSV-007)

### Pre-Handoff Validation
- [ ] Dataset size ≥ 500 examples (post-filtering)
- [ ] Shellcheck pass rate ≥ 95%
- [ ] Zero dangerous commands in final dataset (random sample n=50 verified)
- [ ] Chat template correctness verified (spot-check n=10)
- [ ] Assistant-only masking verified programmatically
- [ ] Provenance.json completeness validated
- [ ] All log files present and non-empty (if violations occurred)

### Spot-Check Validation
- [ ] Overseer loads random 10 examples from train.jsonl
- [ ] Verifies chat template format visually
- [ ] Verifies assistant-only masking (labels[:assistant_idx] all -100)
- [ ] Verifies no dangerous patterns in commands (manual check)

### Handoff Approval
- [ ] All acceptance criteria met
- [ ] PE deliverables complete (`scripts/preprocess_data.py` executable)
- [ ] Documentation updated (README.md mentions data pipeline)
- [ ] Overseer approves transition to Phase 2 (Training)

---

## LINKS
- **Northstar:** `CLI-Tuner_Northstar_FINAL.md#3.3`
- **Index:** `SPECIFICATION_INDEX.md`
- **Previous Phase:** `Phase_0_Setup_SPEC.md`
- **Next Phase:** `Phase_2_Training_SPEC.md` (to be created)

---

**Status:** Ready for Overseer Review  
**Next Step:** Overseer validates acceptance criteria, approves handoff to PE
