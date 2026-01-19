#!/bin/bash
# Phase 3 Dependency Verification Script
# Usage: bash scripts/verify_eval_dependencies.sh

set -e

echo "======================================"
echo "Phase 3 Dependency Verification"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "[1/8] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}[OK]${NC} Python $python_version (>= $required_version)"
else
    echo -e "${RED}[FAIL]${NC} Python $python_version (< $required_version required)"
    exit 1
fi
echo ""

# Check Python packages
echo "[2/8] Checking Python packages..."
packages=(
    "torch"
    "transformers"
    "peft"
    "bitsandbytes"
    "accelerate"
    "datasets"
    "wandb"
    "pydantic"
    "tqdm"
    "yaml:pyyaml"
    "pytest"
)

for pkg in "${packages[@]}"; do
    import_name="${pkg%%:*}"
    pip_name="${pkg#*:}"
    if [ "$import_name" = "$pip_name" ]; then
        pip_name="$import_name"
    fi

    if python -c "import $import_name" 2>/dev/null; then
        version=$(python -c "import $import_name; print($import_name.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}[OK]${NC} $pip_name ($version)"
    else
        echo -e "${RED}[FAIL]${NC} $pip_name NOT INSTALLED"
        echo "   Install: pip install $pip_name"
        exit 1
    fi
done
echo ""

# Check lm-eval (special case)
echo "[3/8] Checking lm-evaluation-harness..."
if python -c "import lm_eval" 2>/dev/null; then
    echo -e "${GREEN}[OK]${NC} lm-evaluation-harness installed"
else
    echo -e "${YELLOW}[WARN]${NC} lm-evaluation-harness NOT installed (optional for Component 3)"
    echo "   Install: pip install lm-evaluation-harness"
fi
echo ""

# Check CUDA availability
echo "[4/8] Checking CUDA..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    cuda_version=$(python -c "import torch; print(torch.version.cuda)")
    echo -e "${GREEN}[OK]${NC} CUDA available: $gpu_name (CUDA $cuda_version)"
else
    echo -e "${YELLOW}[WARN]${NC} CUDA not available - evaluation will run on CPU (VERY SLOW)"
fi
echo ""

# Check shellcheck
echo "[5/8] Checking shellcheck..."
if command -v shellcheck &> /dev/null; then
    shellcheck_version=$(shellcheck --version | grep "version:" | awk '{print $2}')
    echo -e "${GREEN}[OK]${NC} shellcheck $shellcheck_version"
else
    echo -e "${YELLOW}[WARN]${NC} shellcheck not found (syntax validation will be skipped)"
    echo "   Install: apt-get install shellcheck  (Ubuntu/Debian)"
    echo "           brew install shellcheck      (macOS)"
fi
echo ""

# Check git
echo "[6/8] Checking git..."
if command -v git &> /dev/null; then
    git_version=$(git --version | awk '{print $3}')
    echo -e "${GREEN}[OK]${NC} git $git_version"
else
    echo -e "${RED}[FAIL]${NC} git not found (required for provenance tracking)"
    exit 1
fi
echo ""

# Check required files
echo "[7/8] Checking required files..."
required_files=(
    "models/checkpoints/phase2-final/adapter_model.safetensors"
    "models/checkpoints/phase2-final/adapter_config.json"
    "data/processed/test.jsonl"
    "guardrails/patterns.py"
    "configs/evaluation_config.yaml"
    "data/adversarial/dangerous_prompts.jsonl"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        echo -e "${GREEN}[OK]${NC} $file ($size)"
    else
        echo -e "${RED}[FAIL]${NC} $file NOT FOUND"
        if [[ "$file" == "configs/evaluation_config.yaml" ]] || [[ "$file" == "data/adversarial/dangerous_prompts.jsonl" ]]; then
            echo "   This file should be created in Sprint 1"
        else
            echo "   This file should exist from Phase 1-2"
            exit 1
        fi
    fi
done
echo ""

# Check test data integrity
echo "[8/8] Checking test data integrity..."
test_count=$(wc -l < data/processed/test.jsonl 2>/dev/null || echo "0")
if [ "$test_count" -eq 174 ]; then
    echo -e "${GREEN}[OK]${NC} Test data: 174 examples"
else
    echo -e "${RED}[FAIL]${NC} Test data: $test_count examples (expected 174)"
    exit 1
fi
echo ""

# Check dangerous patterns
pattern_count=$(python -c "from guardrails.patterns import DANGEROUS_COMMAND_PATTERNS; print(len(DANGEROUS_COMMAND_PATTERNS))" 2>/dev/null || echo "0")
if [ "$pattern_count" -eq 17 ]; then
    echo -e "${GREEN}[OK]${NC} Dangerous patterns: 17 patterns"
else
    echo -e "${RED}[FAIL]${NC} Dangerous patterns: $pattern_count (expected 17)"
    exit 1
fi
echo ""

echo "======================================"
echo -e "${GREEN}[OK] All dependencies verified${NC}"
echo "======================================"
echo ""
echo "Ready to proceed with Phase 3 evaluation!"
