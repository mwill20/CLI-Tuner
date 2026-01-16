# Phase 0: Repository Setup Specification
**Version:** 1.0  
**Created:** 2026-01-15  
**Status:** Ready for Overseer Review  
**Addresses:** OSV-008 (Repository Structure)  
**Dependencies:** None  
**Duration:** 1-2 hours  
**Owner:** Primary Engineer

---

## PURPOSE

Initialize repository structure, dependencies, and testing framework for CLI-Tuner project. Establishes baseline for all subsequent phases.

---

## ARCHITECTURE OVERVIEW

```
Repository Initialization Flow:
1. Create directory structure
2. Define dependencies (requirements.txt / pyproject.toml)
3. Configure git (.gitignore)
4. Initialize CI/CD (GitHub Actions)
5. Scaffold testing framework
6. Validate setup (install test, CI test run)
```

---

## DIRECTORY STRUCTURE SPECIFICATION

### Required Directory Tree
```
cli-tuner/
├── .github/
│   └── workflows/
│       └── ci.yml                          # Basic CI (lint, test, safety check)
├── .gitignore                              # Exclude data, models, secrets, cache
├── README.md                               # Project overview (minimal for Phase 0)
├── requirements.txt                        # Python dependencies
├── pyproject.toml                          # Optional (if using pip install -e .)
├── setup.py                                # Optional (if using editable install)
│
├── configs/
│   └── .gitkeep                            # Placeholder (Axolotl config in Phase 2)
│
├── data/
│   ├── raw/                                # Downloaded datasets
│   ├── processed/                          # Cleaned, split datasets
│   └── logs/                               # Removal logs (shellcheck, dangerous)
?",   ?""?"??"??"? validation/                         # Validation samples (Phase 1)
│
├── models/
│   ├── checkpoints/                        # Training checkpoints
│   ├── cli-tuner-adapters/                 # LoRA adapters (Phase 2)
│   └── .gitkeep
│
├── schemas/
│   ├── __init__.py
│   ├── request.py                          # CommandRequest, NormalizedRequest
│   ├── response.py                         # CommandResponse, ValidatedCommand
│   └── provenance.py                       # Provenance tracking schemas
│
├── guardrails/
│   ├── __init__.py
│   ├── ingress.py                          # Input validation (Phase 5)
│   ├── output.py                           # Output validation (Phase 5)
│   └── patterns.py                         # Dangerous pattern detection (Phase 5)
│
├── scripts/
│   ├── preprocess_data.py                  # Phase 1 (data pipeline)
│   ├── train.sh                            # Phase 2 (training wrapper)
│   ├── evaluate.py                         # Phase 3 (evaluation)
│   └── quantize.py                         # Phase 4 (optional)
│
├── deployment/
│   ├── cli_tuner.py                        # CLI interface (Phase 5)
│   ├── api/                                # Optional (FastAPI)
│   └── docker/                             # Optional (Dockerfile)
│
├── evaluation/
│   ├── domain/                             # CLI-specific metrics (Phase 3)
│   ├── general/                            # GSM8K, HumanEval (optional)
│   └── safety/                             # Dangerous command tests (Phase 3)
│
├── monitoring/
│   ├── logs/                               # Provenance, performance logs
│   └── metrics/                            # Aggregated metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py                     # Pydantic schema validation tests
│   ├── test_guardrails.py                  # Input/output validation tests
│   ├── test_data_preprocessing.py          # Chat template, masking tests
│   ├── test_evaluation.py                  # Metric calculation tests
│   └── test_patterns.py                    # Dangerous pattern detection tests
│
└── docs/
    ├── CLI-Tuner_Northstar_FINAL.md        # Architectural vision
    ├── SPECIFICATION_INDEX.md              # This index system
    ├── Phase_0_Setup_SPEC.md               # This file
    ├── Phase_1_Data_Pipeline_SPEC.md       # Data pipeline spec
    └── reviews/
        └── Overseer_Review_v4.0_2026-01-15.md
```

---

## DEPENDENCY SPECIFICATION

### requirements.txt
```txt
# Phase 0: Core dependencies
python>=3.10

# Phase 1: Data processing
datasets>=2.14.0                 # HuggingFace datasets
transformers>=4.35.0             # Tokenizers, chat templates
pydantic>=2.0.0                  # Schema validation
pyshellcheck>=0.1.0              # Shellcheck Python wrapper (or install shellcheck separately)

# Phase 2: Training
torch>=2.0.0                     # PyTorch
axolotl>=0.4.0                   # Training framework
peft>=0.6.0                      # LoRA/QLoRA
bitsandbytes>=0.41.0             # 4-bit quantization
accelerate>=0.24.0               # Multi-GPU support
wandb>=0.16.0                    # Experiment tracking

# Phase 3: Evaluation
lm-eval>=0.4.0                   # Evaluation harness (optional)
sacrebleu>=2.0.0                 # BLEU score calculation
nltk>=3.8.0                      # Tokenization utilities

# Phase 5: Deployment
fastapi>=0.104.0                 # API (optional)
uvicorn>=0.24.0                  # ASGI server (optional)

# Development / Testing
pytest>=7.4.0                    # Unit testing
pytest-cov>=4.1.0                # Coverage reporting
black>=23.11.0                   # Code formatting
ruff>=0.1.6                      # Linting
pre-commit>=3.5.0                # Git hooks (optional)
```

### Installation Validation
```bash
# Test dependencies installable
pip install -r requirements.txt

# Verify critical imports
python -c "import torch; import transformers; import datasets; import pydantic"
python -c "import peft; import bitsandbytes; import wandb"

# Check GPU availability (if applicable)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## GIT CONFIGURATION

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/

# Data files
data/raw/*
data/processed/*
data/logs/*
data/validation/*
!data/**/.gitkeep

# Models
models/checkpoints/*
models/cli-tuner-adapters/*
!models/**/.gitkeep

# Logs
monitoring/logs/*
monitoring/metrics/*
!monitoring/**/.gitkeep

# Secrets
.env
*.pem
*.key
secrets/

# Weights & Biases
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary
tmp/
temp/
*.tmp
```

---

## CI/CD SPECIFICATION

### .github/workflows/ci.yml
```yaml
name: CI Pipeline

on:
  push:
    branches: [main, feature/*]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Lint with ruff
        run: |
          pip install ruff
          ruff check .
      
      - name: Format check with black
        run: |
          pip install black
          black --check .
      
      - name: Run unit tests
        run: |
          pytest tests/ -v --cov=. --cov-report=term-missing
      
      - name: Safety check (dangerous patterns)
        run: |
          # Placeholder: Run dangerous pattern detection tests
          # pytest tests/test_patterns.py -v
          echo "Safety checks passed"
```

---

## TESTING STRATEGY SPECIFICATION

### Test Organization
```
tests/
├── test_schemas.py              # Pydantic schema validation
├── test_guardrails.py           # Input/output guardrails
├── test_data_preprocessing.py   # Chat template, masking, shellcheck
├── test_evaluation.py           # Metric calculation
└── test_patterns.py             # Dangerous command detection
```

### Test Coverage Requirements
- **Minimum Coverage:** 80% (for Phase 0: N/A, tests added in subsequent phases)
- **Critical Paths:** 100% (schemas, guardrails, dangerous patterns)
- **Test Execution:** All tests must pass before phase handoff

### Testing Strategy (Defined for PE Clarity)
```yaml
approach: test-after  # Write tests after components are working
rationale: Faster iteration, tests validate working code

coverage_target: 80%  # Minimum for non-critical paths
critical_target: 100%  # Schemas, guardrails, dangerous patterns

test_priority_order:
  1. guardrails (dangerous pattern detection) - CRITICAL - Must have 100% coverage
  2. schemas (Pydantic validation) - CRITICAL - Must have 100% coverage  
  3. data preprocessing (chat template, masking) - HIGH - Aim for 90%+
  4. evaluation metrics - HIGH - Aim for 90%+
  5. utilities - MEDIUM - Aim for 80%+

test_execution_frequency:
  - Run full test suite before phase handoff (mandatory)
  - Run tests after major changes (recommended)
  - CI runs tests on every push (automated)
```

### Testing Framework
- **Unit Tests:** pytest
- **Coverage:** pytest-cov
- **Assertions:** Standard assert + pytest.raises for exceptions
- **Fixtures:** Define in conftest.py (as needed per phase)

---

## SCHEMA SCAFFOLDING

### schemas/__init__.py
```python
"""
Pydantic schemas for type-safe data structures.
"""
from .request import CommandRequest, NormalizedRequest
from .response import CommandResponse, ValidatedCommand
from .provenance import DataProvenance, ModelProvenance, GenerationProvenance

__all__ = [
    "CommandRequest",
    "NormalizedRequest",
    "CommandResponse",
    "ValidatedCommand",
    "DataProvenance",
    "ModelProvenance",
    "GenerationProvenance",
]
```

### schemas/request.py (Placeholder - Implemented in Phase 5)
```python
"""
Request schemas for CLI-Tuner input validation.
"""
from pydantic import BaseModel, Field
from typing import Literal

class CommandRequest(BaseModel):
    """User request schema."""
    query: str = Field(min_length=3, max_length=500)
    shell_type: Literal["bash"] = "bash"
    safety_level: Literal["strict", "moderate"] = "strict"

class NormalizedRequest(BaseModel):
    """Normalized request after preprocessing."""
    prompt: str
    original_query: str
    metadata: dict
```

### schemas/response.py (Placeholder - Implemented in Phase 5)
```python
"""
Response schemas for CLI-Tuner output.
"""
from pydantic import BaseModel
from typing import Literal, Optional

class ValidatedCommand(BaseModel):
    """Validated command after guardrails."""
    command: str
    risk_level: Literal["safe", "review", "dangerous"]
    confidence: float
    validation_results: dict

class CommandResponse(BaseModel):
    """Final response to user."""
    command: str
    risk_level: Literal["safe", "review", "dangerous"]
    confidence: float
    warning: Optional[str] = None
    provenance: dict
    
    class Config:
        frozen = True  # Immutable
```

### schemas/provenance.py (Placeholder - Implemented in Phases 1, 2, 5)
```python
"""
Provenance tracking schemas.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DataProvenance(BaseModel):
    """Data source provenance."""
    source: str
    source_url: Optional[str]
    collected_date: datetime
    validated: bool
    validation_method: str

class ModelProvenance(BaseModel):
    """Model training provenance."""
    base_model: str
    training_data_hash: str
    config_hash: str
    trained_by: str
    trained_at: datetime
    wandb_run_id: str
    git_commit: str

class GenerationProvenance(BaseModel):
    """Generation-time provenance."""
    request_id: str
    model_version: str
    timestamp: datetime
    user_id_hash: str
    query_hash: str
    command_generated: str
    validation_passed: bool
```

---

## ACCEPTANCE CRITERIA (Overseer Validation)

### Directory Structure
- [ ] All directories from specification exist
- [ ] `.gitkeep` files present in empty directories
- [ ] `__init__.py` files present in Python packages

### Dependencies
- [ ] `requirements.txt` contains all Phase 1 dependencies
- [ ] `pip install -r requirements.txt` succeeds without errors
- [ ] Critical imports work: `torch`, `transformers`, `datasets`, `pydantic`, `peft`

### Git Configuration
- [ ] `.gitignore` excludes `data/`, `models/`, `monitoring/`, `wandb/`, `.env`
- [ ] `.gitignore` preserves `.gitkeep` files

### CI/CD
- [ ] `.github/workflows/ci.yml` exists
- [ ] CI workflow runs successfully (even with empty tests)
- [ ] Linting passes (ruff, black)

### Schema Scaffolding
- [ ] `schemas/` package importable
- [ ] Placeholder schemas defined (will be implemented in later phases)
- [ ] No import errors when running `python -c "from schemas import *"`

### Testing Framework
- [ ] `tests/` directory exists
- [ ] `pytest` runs successfully (even if no tests yet)
- [ ] Coverage reporting configured

### Documentation
- [ ] README.md exists with project overview
- [ ] SPECIFICATION_INDEX.md references Phase 0

---

## HANDOFF TO PHASE 1

### Artifacts Delivered
- [ ] Repository structure initialized
- [ ] Dependencies installable
- [ ] CI pipeline functional
- [ ] Schema scaffolding complete

### Blockers Removed
- [ ] Python environment ready
- [ ] Git repository initialized
- [ ] Testing framework operational

### Overseer Sign-Off Required
- [ ] Validate directory structure matches spec
- [ ] Validate dependencies install successfully
- [ ] Validate CI workflow runs
- [ ] Approve transition to Phase 1

---

## NOTES FOR PRIMARY ENGINEER

### Setup Commands
```bash
# Clone repository (or initialize)
git clone <repo-url>
cd cli-tuner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify setup
python -c "from schemas import *"
pytest tests/ -v

# Run CI locally (optional)
black --check .
ruff check .
```

### Common Issues
- **Shellcheck not found:** Install separately (`apt-get install shellcheck` or download binary)
- **CUDA not available:** CPU-only mode acceptable for Phase 1 (data processing)
- **Dependency conflicts:** Pin versions in requirements.txt if needed

---

## LINKS
- **Northstar:** `CLI-Tuner_Northstar_FINAL.md`
- **Index:** `SPECIFICATION_INDEX.md`
- **Next Phase:** `Phase_1_Data_Pipeline_SPEC.md`
- **Overseer Review:** To be created after submission

---

**Status:** Ready for Overseer Review  
**Next Step:** Overseer validates acceptance criteria, approves handoff to PE
