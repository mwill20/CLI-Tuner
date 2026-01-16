# PRIMARY ENGINEER HANDOFF: PHASE 1 DATA PIPELINE
**Date:** 2026-01-15  
**From:** Overseer  
**To:** Primary Engineer  
**Status:** ✅ APPROVED - Ready for Implementation  
**Specification:** docs/Phase_1_Data_Pipeline_SPEC.md  
**Dependencies:** Phase 0 complete (repository initialized)

---

## MISSION

Transform raw HuggingFace dataset into training-ready JSONL files with full security validation and audit trail.

**Input:** `prabhanshubhowal/natural_language_to_linux` (HuggingFace Hub)  
**Output:** `train.jsonl`, `val.jsonl`, `test.jsonl`, `provenance.json`  
**Duration:** 1-2 weeks  
**Critical:** Zero dangerous commands in final dataset

---

## CURRENT EXECUTION DEFAULTS

- Sample run enabled: `SAMPLE_SIZE = 1835` (random sample, `SAMPLE_SEED = 42`)
- Field mapping: `nl_command` → `instruction`, `bash_code` → `output`
- Chat template: Qwen format with `<|im_start|>` / `<|im_end|>`
- Deduplication: full-record fingerprinting before split

---

## IMPLEMENTATION SKELETON

This is your code blueprint. All component details are in [docs/Phase_1_Data_Pipeline_SPEC.md](docs/Phase_1_Data_Pipeline_SPEC.md).

### File Structure
```
scripts/preprocess_data.py  # Main pipeline (you create this)
guardrails/patterns.py      # Dangerous command patterns (from Phase 0)
schemas/dataset.py          # Pydantic models (from Phase 0)
```

### Component Function Signatures

```python
# scripts/preprocess_data.py

from datasets import load_dataset
from transformers import AutoTokenizer
from pydantic import ValidationError
import subprocess
import hashlib
import json

# ===== COMPONENT 1: Dataset Loading =====
def load_and_validate_dataset(dataset_id: str) -> dict:
    """
    Load dataset from HF Hub and validate schema with Pydantic.
    
    Args:
        dataset_id: HuggingFace dataset identifier
        
    Returns:
        dict with keys:
            - valid_examples: list[dict] - Pydantic-validated records
            - schema_violations: list[dict] - Records that failed validation
            - missing_fields: list[dict] - Records with missing required fields
    
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 89-156
    """
    pass

# ===== COMPONENT 2: Shellcheck Validation =====
def check_shellcheck_installed() -> str:
    """
    Pre-flight check: Verify shellcheck is installed.
    
    Returns:
        str: Shellcheck version string
        
    Raises:
        RuntimeError: If shellcheck not found
        
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 235-290
    """
    pass

def filter_invalid_syntax(examples: list[dict]) -> dict:
    """
    Run shellcheck on each command, remove syntax errors.
    
    Args:
        examples: List of validated examples with 'output' field
        
    Returns:
        dict with keys:
            - valid_examples: list[dict] - Commands that passed shellcheck
            - removed_examples: list[dict] - Commands with syntax errors
            - shellcheck_version: str - Version used for validation
            
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 235-290
    """
    pass

# ===== COMPONENT 3: Dangerous Command Filtering =====
def filter_dangerous_commands(examples: list[dict]) -> dict:
    """
    Remove commands matching zero-tolerance dangerous patterns.
    
    Args:
        examples: List of syntax-valid examples
        
    Returns:
        dict with keys:
            - safe_examples: list[dict] - Commands that passed all checks
            - removed_examples: list[dict] - Dangerous commands removed
            - patterns_matched: dict[str, int] - Count per pattern type
            
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 292-356
    """
    pass

def verify_no_dangerous_commands(examples: list[dict]) -> None:
    """
    CRITICAL: Assert no dangerous commands remain before saving.
    
    Args:
        examples: Final dataset about to be saved
        
    Raises:
        RuntimeError: If ANY dangerous command found
        
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 292-356
    """
    pass

# ===== COMPONENT 4: Chat Template Application =====
def apply_chat_template(examples: list[dict]) -> list[dict]:
    """
    Apply Qwen2.5 chat template to all examples.
    
    Format: <|user|>{instruction}<|eot|><|assistant|>{output}<|eot|>
    
    Args:
        examples: List of safe examples with instruction/output fields
        
    Returns:
        list[dict]: Examples with new 'text' field containing formatted chat
        
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 358-406
    """
    pass

# ===== COMPONENT 5: Assistant-Only Masking =====
def tokenize_with_masking(examples: list[dict], tokenizer) -> list[dict]:
    """
    Tokenize and mask user tokens in labels (-100).
    
    Args:
        examples: Chat-formatted examples with 'text' field
        tokenizer: Qwen2.5 tokenizer instance
        
    Returns:
        list[dict]: Examples with input_ids and labels (user masked to -100)
        
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 408-471
    """
    pass

# ===== COMPONENT 6: Train/Val/Test Split =====
def split_dataset(examples: list[dict], seed: int = 42) -> dict:
    """
    Split into 80/10/10 train/val/test with leakage check.
    
    Args:
        examples: Tokenized examples
        seed: Random seed for reproducibility
        
    Returns:
        dict with keys:
            - train: list[dict] - 80% of dataset
            - val: list[dict] - 10% of dataset
            - test: list[dict] - 10% of dataset
            
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 473-519
    """
    pass

# ===== COMPONENT 7: Provenance Recording =====
def record_provenance(stats: dict, splits: dict, output_path: str) -> None:
    """
    Record full audit trail with SHA256 hash.
    
    Required fields (9):
    - source_dataset, source_url, collection_date
    - total_examples_downloaded, final_train_size, final_val_size, final_test_size
    - shellcheck_version, data_hash_sha256
    
    Args:
        stats: Pipeline statistics (counts, versions)
        splits: Train/val/test splits
        output_path: Where to save provenance.json
        
    Spec reference: docs/Phase_1_Data_Pipeline_SPEC.md lines 521-581
    """
    pass

# ===== MAIN PIPELINE =====
def main():
    """
    Execute full 7-component pipeline.
    
    Data flow:
    Raw HF → Pydantic validation → Shellcheck → Dangerous filter 
    → Chat template → Tokenize+mask → Split → Save with provenance
    """
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    stats = {}
    
    # Component 1: Load & validate
    print("Step 1/7: Loading dataset...")
    data = load_and_validate_dataset("prabhanshubhowal/natural_language_to_linux")
    stats["total_downloaded"] = len(data["valid_examples"])
    stats["schema_violations"] = len(data["schema_violations"])
    
    # Component 2: Shellcheck validation
    print("Step 2/7: Running shellcheck...")
    shellcheck_version = check_shellcheck_installed()
    valid_data = filter_invalid_syntax(data["valid_examples"])
    stats["shellcheck_version"] = shellcheck_version
    stats["syntax_invalid"] = len(valid_data["removed_examples"])
    
    # Component 3: Dangerous command filtering
    print("Step 3/7: Filtering dangerous commands...")
    safe_data = filter_dangerous_commands(valid_data["valid_examples"])
    stats["dangerous_removed"] = len(safe_data["removed_examples"])
    stats["patterns_matched"] = safe_data["patterns_matched"]
    
    # Component 4: Chat template
    print("Step 4/7: Applying chat template...")
    formatted_data = apply_chat_template(safe_data["safe_examples"])
    
    # Component 5: Tokenization with masking
    print("Step 5/7: Tokenizing with assistant-only masking...")
    tokenized_data = tokenize_with_masking(formatted_data, tokenizer)
    
    # Component 6: Split dataset
    print("Step 6/7: Splitting train/val/test...")
    splits = split_dataset(tokenized_data, seed=42)
    stats["final_train"] = len(splits["train"])
    stats["final_val"] = len(splits["val"])
    stats["final_test"] = len(splits["test"])
    
    # CRITICAL: Verify no dangerous commands before saving
    verify_no_dangerous_commands(splits["train"] + splits["val"] + splits["test"])
    
    # Component 7: Save outputs with provenance
    print("Step 7/7: Saving outputs...")
    save_jsonl(splits["train"], "data/processed/train.jsonl")
    save_jsonl(splits["val"], "data/processed/val.jsonl")
    save_jsonl(splits["test"], "data/processed/test.jsonl")
    record_provenance(stats, splits, "data/processed/provenance.json")
    
    print(f"✅ Pipeline complete. {stats['final_train']} train, {stats['final_val']} val, {stats['final_test']} test examples.")

# ===== UTILITIES =====
def save_jsonl(data: list[dict], filepath: str) -> None:
    """Save list of dicts to JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()
```

### Data Flow Diagram
```
HuggingFace Hub
    ↓
[Component 1] load_and_validate_dataset()
    ↓ valid_examples (list[dict])
[Component 2] filter_invalid_syntax()
    ↓ valid_examples (list[dict])
[Component 3] filter_dangerous_commands()
    ↓ safe_examples (list[dict])
[Component 4] apply_chat_template()
    ↓ formatted (list[dict] with 'text' field)
[Component 5] tokenize_with_masking()
    ↓ tokenized (list[dict] with input_ids/labels)
[Component 6] split_dataset()
    ↓ splits (dict: train/val/test)
[Component 7] record_provenance() + save_jsonl()
    ↓
data/processed/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    └── provenance.json
```

---

## IMPLEMENTATION ROADMAP

### Week 1: Core Pipeline Development

**Day 1-2: Foundation (Components 1-3)**
```python
# scripts/preprocess_data.py structure

# Component 1: Dataset Loading
def load_and_validate_dataset(dataset_id: str) -> dict:
    """Load from HF Hub, validate schema with Pydantic."""
    # See spec lines 89-156
    pass

# Component 2: Shellcheck Validation  
def filter_invalid_syntax(examples: list) -> dict:
    """Filter commands that fail shellcheck."""
    # See spec lines 235-290
    pass

# Component 3: Dangerous Command Filtering
def filter_dangerous_commands(examples: list) -> dict:
    """Remove commands matching zero-tolerance patterns."""
    # See spec lines 292-356
    pass
```

**Day 3-4: Template & Masking (Components 4-5)**
```python
# Component 4: Chat Template Application
def apply_chat_template(examples: list) -> list:
    """Apply Qwen2.5 chat template to all examples."""
    # See spec lines 358-406
    pass

# Component 5: Assistant-Only Masking
def tokenize_with_masking(examples: list) -> list:
    """Tokenize and mask user tokens (-100 in labels)."""
    # See spec lines 408-471
    pass
```

**Day 5-6: Split & Provenance (Components 6-7)**
```python
# Component 6: Train/Val/Test Split
def split_dataset(examples: list, seed: int = 42) -> dict:
    """80/10/10 split with leakage check."""
    # See spec lines 473-519
    pass

# Component 7: Provenance Recording
def record_provenance(stats: dict, final_splits: dict) -> None:
    """Record full audit trail with SHA256 hashes."""
    # See spec lines 521-581
    pass
```

**Day 7: Integration & Testing**
```python
# Main pipeline
def main():
    # 1. Load & validate
    data = load_and_validate_dataset("prabhanshubhowal/natural_language_to_linux")
    
    # 2. Shellcheck validation
    valid_data = filter_invalid_syntax(data["valid_examples"])
    
    # 3. Dangerous filtering
    safe_data = filter_dangerous_commands(valid_data["valid_examples"])
    
    # 4. Chat template
    formatted_data = apply_chat_template(safe_data["safe_examples"])
    
    # 5. Masking
    tokenized_data = tokenize_with_masking(formatted_data)
    
    # 6. Split
    splits = split_dataset(tokenized_data)
    
    # 7. Provenance
    record_provenance(stats, splits)
    
    # 8. Save outputs
    save_jsonl(splits["train"], "data/processed/train.jsonl")
    save_jsonl(splits["val"], "data/processed/val.jsonl")
    save_jsonl(splits["test"], "data/processed/test.jsonl")
```

---

### Week 2: Validation & Handoff

**Day 8-9: Self-Validation**
```bash
# Run pipeline
python scripts/preprocess_data.py

# Verify outputs
ls -lh data/processed/  # Should see train/val/test.jsonl + provenance.json
ls -lh data/logs/       # Should see 4 log files

# Check provenance
python -c "import json; print(json.load(open('data/processed/provenance.json')))"
```

**Day 10: Overseer Validation Prep**
```python
# Generate validation artifacts
python scripts/generate_validation_sample.py  # Creates random sample for Overseer

# Outputs:
# - data/validation/random_sample_50.jsonl (for dangerous pattern check)
# - data/validation/chat_template_sample_10.txt (for manual review)
```

**Day 11: Submit to Overseer**
- Submit completion notice
- Provide validation artifacts
- Await Overseer review (6 validation checks)

---

## ACCEPTANCE CRITERIA (Self-Check Before Submission)

### Data Quality
- [ ] Dataset size ≥ 500 examples (post-filtering)
- [ ] Shellcheck pass rate ≥ 95%
- [ ] Zero dangerous commands (run verification script)
- [ ] Chat template tokens present in all examples (`<|user|>`, `<|assistant|>`, `<|eot|>`)
- [ ] Assistant-only masking verified (user tokens = -100)

### File Outputs
- [ ] `data/processed/train.jsonl` exists (80% of dataset)
- [ ] `data/processed/val.jsonl` exists (10% of dataset)
- [ ] `data/processed/test.jsonl` exists (10% of dataset)
- [ ] `data/processed/provenance.json` exists (9 required fields)
- [ ] `data/logs/schema_violations.jsonl` exists (if violations occurred)
- [ ] `data/logs/removed_invalid_syntax.jsonl` exists (if syntax errors found)
- [ ] `data/logs/removed_dangerous.jsonl` exists (if dangerous commands found)

### Code Quality
- [ ] `scripts/preprocess_data.py` executable (`python scripts/preprocess_data.py`)
- [ ] No hardcoded paths (use `pathlib` or relative paths)
- [ ] Error handling present (try/except for HF download, shellcheck unavailable)
- [ ] Logging to console (progress updates)
- [ ] Docstrings on all functions

---

## CRITICAL REQUIREMENTS (MUST NOT SKIP)

### 1. Shellcheck Pre-Flight Check
```python
# REQUIRED: Run before processing any commands
def check_shellcheck_installed() -> str:
    """Verify shellcheck is installed."""
    try:
        result = subprocess.run(["shellcheck", "--version"], capture_output=True, timeout=5)
        return result.stdout.decode()
    except FileNotFoundError:
        raise RuntimeError("Shellcheck not installed. Install: apt-get install shellcheck")
```

### 2. Dangerous Pattern Verification
```python
# REQUIRED: Run after filtering, before saving
def verify_no_dangerous_commands(examples: list) -> None:
    """Assert no dangerous commands remain."""
    for example in examples:
        check = is_dangerous_command(example["output"])
        if check["is_dangerous"]:
            raise RuntimeError(f"CRITICAL: Dangerous command slipped through: {example['output']}")
```

### 3. Provenance Completeness
```python
# REQUIRED: All 9 fields must be present
provenance_required_fields = [
    "source_dataset",
    "source_url",
    "collection_date",
    "total_examples_downloaded",
    "final_train_size",
    "final_val_size", 
    "final_test_size",
    "shellcheck_version",
    "data_hash_sha256"
]
```

---

## COMMON PITFALLS & SOLUTIONS

### Pitfall 1: Shellcheck Not Installed
**Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'shellcheck'`  
**Solution:** Install shellcheck before running pipeline
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install shellcheck

# macOS
brew install shellcheck

# Windows (via Scoop)
scoop install shellcheck
```

### Pitfall 2: Dataset Too Small After Filtering
**Symptom:** Only 300 examples after filtering dangerous commands  
**Solution:** Log warning but continue (training may be suboptimal but not blocked)
```python
if len(final_dataset) < 500:
    logging.warning(f"Dataset size {len(final_dataset)} < 500. Training may be suboptimal.")
    # Continue anyway
```

### Pitfall 3: Chat Template Mismatch
**Symptom:** Missing `<|assistant|>` token in formatted text  
**Solution:** Verify tokenizer chat template before processing all examples
```python
# Test on one example first
test_example = examples[0]
formatted = tokenizer.apply_chat_template([{"role": "user", "content": test_example["instruction"]}], tokenize=False)
assert "<|user|>" in formatted, "Chat template broken - user token missing"
```

### Pitfall 4: Dangerous Command Slip
**Symptom:** Overseer validation finds dangerous command in final dataset  
**Solution:** Run verification function before saving
```python
# ALWAYS run this before saving
verify_no_dangerous_commands(final_dataset)
```

---

## OVERSEER VALIDATION CHECKLIST (What I'll Check)

When you submit, I will validate:

1. **Provenance Completeness**
   - [ ] `provenance.json` exists
   - [ ] Contains all 9 required fields
   - [ ] Data hash is 64-char hexadecimal (SHA256)

2. **Chat Template Correctness**
   - [ ] Spot-check 10 random examples from `train.jsonl`
   - [ ] Verify `<|user|>`, `<|assistant|>`, `<|eot|>` tokens present
   - [ ] No raw placeholders (`{instruction}`, `{output}`)

3. **Dangerous Pattern Verification**
   - [ ] Load random sample (n=50) from `train.jsonl`
   - [ ] Run dangerous pattern detection
   - [ ] Assert zero matches

4. **Dataset Size Verification**
   - [ ] Train + val + test sizes match `provenance.json`
   - [ ] Total ≥ 500 examples

5. **Shellcheck Version**
   - [ ] Version recorded in `provenance.json`
   - [ ] Version ≥ 0.9.0

6. **Data Hash Verification**
   - [ ] Recompute SHA256 hash of `train.jsonl`
   - [ ] Compare against `provenance.json` hash

**If all 6 checks pass:** ✅ APPROVED → Hand off to Architect for Phase 2 spec  
**If any check fails:** 🔄 CHANGES REQUESTED → Fix and re-submit

---

## HANDOFF COMPLETE

**Next Steps:**
1. Implement `scripts/preprocess_data.py` per specification
2. Run pipeline and generate outputs
3. Self-validate against acceptance criteria
4. Submit to Overseer for final validation

**Questions?** Refer to `docs/Phase_1_Data_Pipeline_SPEC.md` for detailed specifications.

**Blockers?** Contact Overseer for clarification (specification questions only, not code debugging).

---

**Overseer Signature:** ✅ Phase 1 Approved for PE Implementation  
**Date:** 2026-01-15  
**Specification Version:** docs/Phase_1_Data_Pipeline_SPEC.md v1.0  
**Expected Completion:** Week of 2026-01-22 to 2026-01-29
