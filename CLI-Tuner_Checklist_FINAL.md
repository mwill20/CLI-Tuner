# CLI-Tuner Implementation Checklist
## Based on Northstar v4.0 — Production Security Architecture

**Version:** 1.0  
**Date:** 2025-01-15  
**Purpose:** Track implementation progress against Northstar requirements

---

## Progress Summary

**Total Items:** 120  
**Completed:** 0/120 ☐  
**Target Timeline:** 5-6 weeks (solo, part-time)

### By Phase:
- Phase 1 (Data): 0/15 ☐
- Phase 2 (Training): 0/18 ☐
- Phase 3 (Evaluation): 0/15 ☐
- Phase 4 (Quantization): 0/8 ☐
- Phase 5 (Deployment): 0/32 ☐
- Phase 6 (Documentation): 0/12 ☐
- Phase 7 (Submission): 0/8 ☐
- Operational: 0/12 ☐

---

## PHASE 1: DATA PIPELINE (Weeks 1-2)

### Dataset Acquisition & Validation
- [ ] Load `prabhanshubhowal/natural_language_to_linux` from HuggingFace Hub
- [ ] Verify dataset schema (instruction, input, output fields)
- [ ] Count total examples (target: 500-10k)
- [ ] Check for missing/null values

### Data Cleaning & Validation
- [ ] Install shellcheck for syntax validation
- [ ] Run shellcheck on all output commands
- [ ] Filter invalid commands (syntax errors)
- [ ] Remove duplicate examples (exact matches)

### Dangerous Command Filtering
- [ ] Implement zero-tolerance pattern detection
- [ ] Filter commands matching dangerous patterns (rm -rf /, dd, etc.)
- [ ] Log filtered commands for review
- [ ] Verify no dangerous commands in final dataset

### Chat Template Formatting
- [ ] Apply Qwen chat template to all examples
- [ ] Verify format: `<|user|>...<|eot|><|assistant|>...<|eot|>`
- [ ] Test assistant-only masking (user tokens = -100)

### Train/Val/Test Split
- [ ] Split dataset: 80% train, 10% val, 10% test
- [ ] Verify no data leakage between splits
- [ ] Save splits to `data/processed/`

### Data Provenance
- [ ] Record source (HuggingFace Hub URL)
- [ ] Record collection date
- [ ] Record validation method (shellcheck)
- [ ] Record dangerous command filtering applied
- [ ] Save provenance.json

### HuggingFace Hub Upload
- [ ] Create dataset repository on HuggingFace Hub
- [ ] Upload train/val/test splits
- [ ] Create dataset card (README.md)
- [ ] Publish dataset (public or private)

**Phase 1 Success Gate:** Clean, validated dataset ready for training

---

## PHASE 2: TRAINING (Weeks 2-3)

### Environment Setup
- [ ] Install dependencies (transformers, peft, bitsandbytes, axolotl)
- [ ] Verify GPU availability (24GB VRAM minimum)
- [ ] Test QLoRA loading (load base model in 4-bit)

### Weights & Biases Setup
- [ ] Create W&B account (if needed)
- [ ] Create project: `cli-tuner`
- [ ] Test W&B logging (simple run)

### Axolotl Configuration
- [ ] Create `configs/axolotl.yaml`
- [ ] Set base model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- [ ] Set LoRA config: r=8, alpha=16, target=[q_proj, v_proj]
- [ ] Set quantization: load_in_4bit=true, nf4, double_quant
- [ ] Set training: lr=2e-4, epochs=3, batch_size=4, grad_accum=4
- [ ] Set W&B integration (project, entity)

### Training Execution
- [ ] Load dataset (train/val splits)
- [ ] Initialize model with QLoRA
- [ ] Verify VRAM usage (~6-8GB)
- [ ] Start training (log to W&B)
- [ ] Monitor training loss (should decrease steadily)
- [ ] Monitor validation loss (should track training)
- [ ] Save checkpoints every 500 steps

### Training Completion
- [ ] Training completes successfully (3 epochs)
- [ ] LoRA adapters saved to `models/cli-tuner-adapters/`
- [ ] W&B run logged (full metrics, config)
- [ ] Adapter size verified (~10-50MB)

### Model Provenance
- [ ] Record base model (name, version)
- [ ] Record training data hash (SHA256)
- [ ] Record Axolotl config hash (SHA256)
- [ ] Record trained_by (username)
- [ ] Record trained_at (timestamp)
- [ ] Record W&B run ID
- [ ] Record Git commit hash
- [ ] Save to `models/provenance.json`

**Phase 2 Success Gate:** Trained model with improving loss curve, adapters saved

---

## PHASE 3: EVALUATION (Week 3)

### Domain Metrics (CLI-Specific)
- [ ] Implement Exact Match (EM) calculator
- [ ] Implement Command BLEU scorer
- [ ] Implement syntax validator (shellcheck integration)
- [ ] Implement command-only rate checker (no markdown, no explanations)
- [ ] Run evaluation on test set
- [ ] Calculate metrics: EM ≥ 85%, Command-Only ≥ 98%, Syntax ≥ 95%

### General Benchmarks (Optional)
- [ ] Install lm-evaluation-harness
- [ ] Run GSM8K (math reasoning)
- [ ] Run HumanEval (code generation)
- [ ] Verify no drop >5 points vs base model

### Safety Benchmark (Critical)
- [ ] Create safety test set (100 dangerous commands)
- [ ] Categories: file destruction, fork bombs, privilege escalation, remote execution
- [ ] Run safety benchmark (all commands should be blocked by output validation)
- [ ] Verify TPR = 100% (all dangerous commands detected)
- [ ] Verify FPR = 0% (no benign commands flagged)

### Evaluation Report
- [ ] Generate evaluation report (domain + general + safety metrics)
- [ ] Save to `evaluation/results.md`
- [ ] Include example outputs (good and bad)
- [ ] Document failure modes

**Phase 3 Success Gate:** Metrics meet thresholds (EM ≥85%, Safety 100%)

---

## PHASE 4: QUANTIZATION (Week 4)

### GGUF Quantization
- [ ] Install llama.cpp
- [ ] Convert model to GGUF format
- [ ] Quantize to 4-bit (Q4_K_M)
- [ ] Quantized model size verified (~4GB)

### Quality Validation
- [ ] Re-run domain metrics on quantized model
- [ ] Compare: FP16 vs 4-bit
- [ ] Verify quality degradation <5%
- [ ] Document tradeoffs (quality vs VRAM)

### Performance Benchmarking
- [ ] Measure inference latency (ms per token)
- [ ] Measure VRAM usage (GB)
- [ ] Measure throughput (tokens/second)
- [ ] Save benchmarks to `evaluation/quantization.md`

**Phase 4 Success Gate:** Quantized model with <5% quality loss

---

## PHASE 5: DEPLOYMENT (Weeks 4-5)

### Security Layer: Schemas
- [ ] Define `CommandRequest` schema (Pydantic)
- [ ] Define `NormalizedRequest` schema
- [ ] Define `RawModelOutput` schema
- [ ] Define `ValidatedCommand` schema
- [ ] Define `CommandResponse` schema
- [ ] Define `ProvenanceRecord` schema
- [ ] Unit test all schemas (validation, immutability)

### Security Layer: Guardrails
- [ ] Implement ingress validation (Boundary #1)
  - Schema validation, length limits, injection detection
- [ ] Implement pre-processing (Boundary #2)
  - Normalization, sanitization, chat template
- [ ] Implement output validation (Boundary #4) ← CRITICAL
  - Dangerous pattern detection (zero-tolerance list)
  - Length validation (3-200 chars)
  - Format validation (no markdown)
  - Policy compliance
- [ ] Implement egress formatting (Boundary #5)
  - Structured response, risk level, provenance
- [ ] Unit test all guardrails (positive and negative cases)

### Security Layer: Dangerous Patterns
- [ ] Define zero-tolerance pattern list
- [ ] Patterns: rm -rf /, fork bomb, dd, chmod 777, curl|sh, etc.
- [ ] Implement pattern matching (regex-based)
- [ ] Test on safety benchmark (100% detection)

### Security Layer: Provenance
- [ ] Implement generation provenance logging
- [ ] Log: request_id, timestamp, user_hash, query_hash, command, validation
- [ ] Save to `monitoring/logs/provenance.jsonl`

### Operational Layer: Monitoring
- [ ] Implement performance metrics logging
- [ ] Metrics: latency, validation_pass_rate, risk_level_distribution
- [ ] Save to `monitoring/metrics/performance.jsonl`
- [ ] Implement validation error logging
- [ ] Log: request_id, failure_reason, pattern_matched

### Operational Layer: Feedback Loop
- [ ] Implement automated feedback logging (every validation error)
- [ ] Implement user reporting mechanism (CLI flag: --report-issue)
- [ ] Create feedback database (`monitoring/feedback.jsonl`)

### Deployment: CLI Interface
- [ ] Create `cli_tuner.py` script
- [ ] Implement: `cli_tuner generate "<query>"`
- [ ] Implement: `cli_tuner generate --report-issue`
- [ ] Test 5-line usage example

### Deployment: Docker (Optional)
- [ ] Create Dockerfile
- [ ] Build Docker image
- [ ] Test inference inside container
- [ ] Image size verified (<10GB)

### Deployment: Health Check
- [ ] Implement `/health` endpoint (if API)
- [ ] Returns: status, model_version, uptime

### Deployment: CI/CD
- [ ] Create `.github/workflows/ci.yml`
- [ ] Jobs: lint, unit tests, safety benchmark
- [ ] Test pipeline on push

**Phase 5 Success Gate:** Working inference system with full security validation

---

## PHASE 6: DOCUMENTATION (Week 5)

### Model Card (HuggingFace Format)
- [ ] Section: Model Description
- [ ] Section: Training Procedure
- [ ] Section: Evaluation Results
- [ ] Section: Intended Use
- [ ] Section: Limitations
- [ ] Section: Ethical Considerations
- [ ] Section: Security Considerations (trust boundaries, guardrails)
- [ ] Publish to HuggingFace Hub

### Repository Documentation
- [ ] README.md (project overview, quickstart, installation)
- [ ] ARCHITECTURE.md (system design, trust boundaries, security)
- [ ] SECURITY.md (threat model, guardrails, provenance)
- [ ] EVALUATION.md (metrics, benchmarks, results)

**Phase 6 Success Gate:** Complete, professional documentation

---

## PHASE 7: RT SUBMISSION (Week 5)

### Final QA
- [ ] All RT requirements met (fine-tuning, evaluation, deployment)
- [ ] All Security requirements met (guardrails, provenance, validation)
- [ ] Model published on HuggingFace Hub
- [ ] Documentation complete
- [ ] Repository clean (no secrets, no temp files)

### Submission Deliverables
- [ ] Model link (HuggingFace Hub)
- [ ] Model card (complete)
- [ ] Evaluation report
- [ ] Axolotl config (reproducible)
- [ ] GitHub repository (public or private)
- [ ] W&B run link
- [ ] Deployment guide

**Phase 7 Success Gate:** RT submission complete

---

## OPERATIONAL MATURITY (Ongoing)

### Experiment Tracking
- [ ] W&B project active (all runs logged)
- [ ] Metrics tracked: loss, LR, GPU util, eval results
- [ ] Run IDs recorded in provenance

### Model Versioning
- [ ] Git tags created (v1.0.0, v1.1.0, etc.)
- [ ] HuggingFace Hub versioned (cli-tuner-v1.0.0)
- [ ] Semantic versioning followed

### Performance Monitoring
- [ ] Metrics aggregated weekly (latency, validation rates)
- [ ] Dashboard created (W&B or Grafana - optional)
- [ ] Drift detection active (weekly eval on held-out set)

### Incident Response
- [ ] Playbooks documented (dangerous command slip, model degradation)
- [ ] Rollback procedure tested (blue-green deployment)
- [ ] Post-mortem template created

### Continuous Improvement
- [ ] Feedback loop operational (validation errors logged)
- [ ] Retraining triggers defined (drift >10%, new patterns)
- [ ] Quarterly retraining scheduled

### Deployment Management
- [ ] Blue-green deployment implemented
- [ ] Rollback tested (< 5 minutes)
- [ ] CI/CD pipeline functional

**Operational Success Gate:** All monitoring, versioning, and improvement processes operational

---

## QUALITY THRESHOLDS

### RT Requirements
- ✅ Exact Match ≥ 85%
- ✅ Command-Only ≥ 98%
- ✅ Syntax Validity ≥ 95%
- ✅ No drop >5 points on general benchmarks

### Security Requirements
- ✅ TPR = 100% (all dangerous commands detected)
- ✅ FPR = 0% (no false positives on benign commands)
- ✅ All trust boundaries implemented (6 validation points)
- ✅ Provenance tracked (data → model → generation)

### Operational Requirements
- ✅ W&B logging functional
- ✅ Model versioning active (HuggingFace + Git)
- ✅ Performance monitoring operational
- ✅ Feedback loop functional

---

## COMPLETION TRACKING

**Current Phase:** Not Started  
**Days Elapsed:** 0  
**Items Completed Today:** 0

**Update this section daily to track progress.**

---

**Last Updated:** 2025-01-15  
**Next Review:** After Phase 1 completion
