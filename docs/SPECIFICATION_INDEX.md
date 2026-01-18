# CLI-Tuner Specification Index
**Version:** 1.2  
**Last Updated:** 2026-01-15  
**Purpose:** Track implementation specifications derived from Northstar v4.0  
**Maintainer:** Architect  
**Reviewers:** Overseer (validation), PE (implementation)

---

## DOCUMENT HIERARCHY

```
docs/
  CLI-Tuner_Northstar_FINAL.md          # v4.0 - Architectural vision (high-level)
  SPECIFICATION_INDEX.md                # THIS FILE - Issue tracking + phase mapping
  Phase_0_Setup_SPEC.md                 # Repository initialization
  Phase_1_Data_Pipeline_SPEC.md         # Data preprocessing implementation
  Phase_2_Training_SPEC.md              # Training implementation (spec ready)
  Phase_3_Evaluation_SPEC.md            # Evaluation implementation (TBD)
  Phase_4_Quantization_SPEC.md          # Quantization implementation (TBD - optional)
  Phase_5_Deployment_SPEC.md            # Deployment implementation (TBD)
  Phase_6_Documentation_SPEC.md         # Documentation implementation (TBD)
  Phase_7_RT_Submission_SPEC.md         # Submission checklist (TBD)
  reviews/
    Overseer_Review_v4.0_2026-01-15.md  # Archived feedback
  lessons/
    Lesson_01_Phase1_Data_Pipeline.md
```

---

## OVERSEER ISSUE TRACKING

| Issue ID | Description | Severity | Assigned Phase | Spec Status | Location | Accepted By |
|----------|-------------|----------|----------------|-------------|----------|-------------|
| OSV-001 | Trust boundary validation behavior (schema validation, injection handling, error responses) | Critical | Phase 1 | ✅ Complete | `Phase_1_Data_Pipeline_SPEC.md#L89-156` | Overseer (2026-01-15) |
| OSV-002 | Output guardrails pattern matching (performance, execution order, evasion resistance) | Critical | Phase 5 | ⏳ Not Started | TBD | N/A |
| OSV-003 | Data preprocessing verification (chat template, masking, shellcheck, dangerous filtering) | Critical | Phase 1 | ✅ Complete | `Phase_1_Data_Pipeline_SPEC.md#L158-298` | Overseer (2026-01-15) |
| OSV-004 | Evaluation metric edge cases (whitespace normalization, case sensitivity, rounding rules) | Critical | Phase 3 | ⏳ Not Started | TBD | N/A |
| OSV-005 | Quantization quality degradation (metric definition, measurement, rollback criteria) | Important | Phase 4 | ⏸️ Optional | TBD (post-RT v1.0.0) | N/A |
| OSV-006 | Monitoring implementation (logging format, aggregation, storage, alerting) | Important | Phase 7 | ⏸️ Optional | TBD (post-RT v1.0.0) | N/A |
| OSV-007 | Phase handoff criteria (dependencies, artifacts, success gates, Overseer validation) | Critical | All Phases | 🔄 In Progress | Each phase spec | Overseer (P0/P1: 2026-01-15) |
| OSV-008 | Repository structure (CI/CD, tests, schemas directory) | Important | Phase 0 | ✅ Complete | `Phase_0_Setup_SPEC.md#L45-142` | Overseer (2026-01-15) |

---

## PHASE SPECIFICATION STATUS

### Phase 0: Repository Setup
- **Status:** Implemented (L4)  
- **Spec Location:** `Phase_0_Setup_SPEC.md`  
- **Addresses:** OSV-008 (repository structure, testing strategy)  
- **Dependencies:** None  
- **Duration:** 1-2 hours  
- **Artifacts Produced:**
  - `.gitignore`, `requirements.txt`, `pyproject.toml`
  - `tests/` directory structure
  - `schemas/` directory structure
  - `.github/workflows/ci.yml` (basic)
  - Directory scaffolding (`data/`, `models/`, `guardrails/`, etc.)
- **Success Gate:**
  - [ ] Directory structure matches specification
  - [ ] `requirements.txt` contains all Phase 1 dependencies
  - [ ] CI workflow runs successfully (even if tests are empty)
  - [ ] `.gitignore` excludes data files, model checkpoints, secrets
- **Handoff to Phase 1:** Repository initialized, dependencies installable

---

### Phase 1: Data Pipeline
- **Status:** ✅ Complete (L5) - All validations passed  
- **Spec Location:** `Phase_1_Data_Pipeline_SPEC.md`  
- **Addresses:** OSV-001 (schema validation), OSV-003 (preprocessing verification), OSV-007 (Phase 1 handoff)  
- **Dependencies:** Phase 0 complete  
- **Duration:** 1-2 weeks  
- **Artifacts Produced:**
  - `data/processed/train.jsonl` (1,388 examples with chat template applied)
  - `data/processed/val.jsonl` (173 examples)
  - `data/processed/test.jsonl` (174 examples)
  - `data/processed/provenance.json` (full audit trail)
  - `data/logs/removed_invalid_syntax.jsonl` (42 shellcheck failures)
  - `data/logs/removed_dangerous.jsonl` (0 dangerous commands)
  - `scripts/preprocess_data.py` (603 lines, executable script)
  - `scripts/generate_validation_sample.py` (73 lines, validation artifacts)
  - `data/validation/random_sample_50.jsonl` (dangerous pattern verification)
  - `data/validation/chat_template_sample_10.txt` (manual review sample)
  - `docs/lessons/Lesson_01_Phase1_Data_Pipeline.md` (816 lines, educational content)
- **Success Gate:**
  - [x] Dataset size ≥ 500 examples (1,735 total post-filtering) ✅
  - [x] Shellcheck pass rate ≥ 95% (97.71% actual) ✅
  - [x] Zero dangerous commands in final dataset (0 verified) ✅
  - [x] Chat template correctness verified (Qwen im_start/im_end format) ✅
  - [x] Assistant-only masking verified programmatically ✅
  - [x] Provenance.json completeness validated by Overseer ✅
  - [x] Lesson content validated ?
  - [x] README accuracy verified ✅
- **Handoff to Phase 2:** Clean, validated dataset ready for training
- **Overseer Corrections Applied:**
  - Field mapping added (nl_command -> instruction, bash_code -> output)
  - Chat template verification updated for Qwen im_start/im_end format
  - Assistant-only masking aligned to Qwen assistant marker
  - Deduplication added before split to prevent leakage
  - save_jsonl argument order corrected
  - UTC timestamp warning fixed (datetime.now(timezone.utc))
  - Random sampling with SAMPLE_SEED=42 for reproducibility
- **Final Dataset Stats:**
  - Total downloaded: 18,357 examples
  - Sampled (10%): 1,835 examples (seed=42)
  - Shellcheck validated: 1,793 passed / 42 failed (97.71%)
  - Dangerous filtering: 0 removed
  - Deduplication: 58 duplicates removed
  - Final split: 1,388 train / 173 val / 174 test (1,735 total)

---

### Phase 2: Training Setup & Fine-Tuning
- **Status:** 📋 Ready for Implementation (Spec Complete)  
- **Spec Location:** `docs/Phase_2_Training_SPEC.md`  
- **Addresses:** OSV-007 (Phase 2 handoff), QLoRA training within VRAM constraints  
- **Dependencies:** Phase 1 complete (dataset validated)  
- **Duration:** 1-2 weeks (implementation + initial training runs)  
- **Artifacts Prepared (Pre-training):**
  - `configs/axolotl_config.yaml` (main training configuration)
  - `configs/axolotl_smoke_test.yaml` (10-example validation config)
  - `.env.example` (W&B API key template)
- **Artifacts to Produce (Post-training):**
  - `models/checkpoints/phase2-final/` (LoRA adapters)
  - W&B experiment logs (loss curves, metrics, hyperparameters)
  - `docs/lessons/Lesson_02_Phase2_Training.md` (deferred to after training)
- **Success Gate:**
  - [ ] Smoke test completes without OOM (10 examples, 10 steps)
  - [ ] Full training completes (500 steps, ~1.4 epochs)
  - [ ] Loss converges (validation loss decreases)
  - [ ] Model generates valid Bash commands (syntactically correct)
  - [ ] Checkpoint files saved (<200MB per checkpoint)
  - [ ] W&B run logged with full provenance
  - [ ] GPU memory usage < 12GB (well under 24GB limit)
- **Handoff to Phase 3:** Fine-tuned LoRA adapters ready for evaluation
- **Overseer Validations:**
  - Memory budget verified: ~10GB total (fits in 24GB GPU with 14GB headroom) ✅
  - Example count corrected: 1,735 total (Architect claimed 1,746) ✅
  - Steps per epoch corrected: 347 steps (Architect claimed 437) ✅
  - Validation strategy clarified: Use separate val.jsonl (not split from train) ✅
  - LoRA target modules verified for Qwen2.5 architecture ✅
- **Recommendations:**
  - Start with r=8, extend to r=16 if model underfits
  - Start with 500 steps, extend to 1,000 if loss still decreasing
  - Save checkpoints every 100 steps for resumption safety
  - Monitor W&B dashboard for early stopping signal
- **User Actions Required:**
  - Create W&B account and get API key
  - Install Axolotl: `pip install axolotl`
  - Create `.env` file from `.env.example`
  - (Optional) Set up RunPod if using cloud GPU
- **Latest Execution Summary (10% sample, seed=42):**
  - Total downloaded: 18357
  - Sampled: 1835
  - Schema violations: 0
  - Missing fields: 0
  - Shellcheck: 1793/1835 passed (97.71%)
  - Invalid syntax removed: 42
  - Dangerous commands removed: 0
  - Duplicates removed: 58
  - Final split: 1388 train / 173 val / 174 test (1735 total)

---

### Phase 3: Evaluation
- **Status:** ⏳ Awaiting Phase 2 Completion  
- **Spec Location:** TBD  
- **Addresses:** OSV-002 (output guardrails), OSV-004 (metric edge cases), OSV-007 (Phase 3 handoff)  
- **Dependencies:** Phase 2 complete, Overseer approved  
- **Duration:** 3-5 days  
- **Artifacts Expected:**
  - `evaluation/RESULTS.md`
  - `evaluation/metrics.json`
  - `evaluation/failures.jsonl`
- **Success Gate Preview:**
  - Exact Match ≥ 85%
  - Command-Only Rate ≥ 98%
  - Safety Test: 100% dangerous commands blocked

---

### Phase 4: Quantization (OPTIONAL)
- **Status:** ⏸️ Deferred (Decision Point: Post-Phase 3)  
- **Spec Location:** TBD (only if pursuing quantization)  
- **Addresses:** OSV-005 (quality degradation measurement)  
- **Dependencies:** Phase 3 complete, decision to quantize  
- **Duration:** 2-3 days  
- **Decision Criteria:**
  - Required for RT submission? (Clarify with RT requirements)
  - Time available before deadline?
  - Performance requirements justify quantization?

---

### Phase 5: Deployment
- **Status:** ⏳ Awaiting Phase 3 Completion  
- **Spec Location:** TBD  
- **Addresses:** OSV-002 (output guardrails implementation)  
- **Dependencies:** Phase 3 complete (Phase 4 optional)  
- **Duration:** 1 week  
- **Artifacts Expected:**
  - `deployment/cli_tuner.py` (CLI interface)
  - `guardrails/` modules (input/output validation)
  - `schemas/` Pydantic models
  - 5-line inference example working

---

### Phase 6: Documentation
- **Status:** ⏳ Awaiting Phase 5 Completion  
- **Spec Location:** TBD  
- **Dependencies:** Phase 5 complete  
- **Duration:** 3-5 days  
- **Artifacts Expected:**
  - Model card (HuggingFace format)
  - README.md
  - ARCHITECTURE.md
  - EVALUATION.md

---

### Phase 7: RT Submission
- **Status:** ⏳ Awaiting Phase 6 Completion  
- **Spec Location:** TBD (submission checklist)  
- **Addresses:** OSV-006 (monitoring - optional for v1.0.0)  
- **Dependencies:** All required phases complete  
- **Duration:** 1-2 days (final QA)  
- **Deliverables:**
  - HuggingFace Hub model published
  - GitHub repository public/private
  - W&B run link
  - Evaluation report

---

## SPECIFICATION MATURITY LEVELS

| Level | Name | Description | Owner |
|-------|------|-------------|-------|
| **L0** | Placeholder | Issue identified, not yet designed | Overseer (issue created) |
| **L1** | Architecture | High-level design exists in Northstar | Architect (vision) |
| **L2** | Implementation Spec | YAML-style deterministic requirements written | Architect (detailed spec) |
| **L3** | PE-Ready | Overseer approved, handed off to Primary Engineer | Overseer (validated) |
| **L4** | Implemented | Code exists, tests passing | Primary Engineer (built) |
| **L5** | Validated | Meets acceptance criteria, Overseer sign-off | Overseer (verified) |

---

## CURRENT PRIORITIES (2026-01-15)

### Immediate (This Week)
1. ? **Overseer Review:** Phase 2 Training Spec (APPROVED 2026-01-15)
2. ?? **PE Implementation:** Phase 2 smoke test prep (Axolotl install, W&B setup, smoke datasets)
3. ? **PE Execution:** Phase 2 smoke test run (10 steps)

### Short-Term (Weeks 1-2)
1. **PE Execution:** Phase 2 full training (500 steps)
2. **Overseer Review:** Phase 2 smoke test + training results
3. **Architect Design:** Phase 3 Evaluation Spec

### Medium-Term (Weeks 3-4)
1. **PE Implementation:** Phase 3 Evaluation
2. **Decision Point:** Quantization required?

### Long-Term (Week 5+)
1. **PE Implementation:** Phase 5 Deployment
2. **Architect Design:** Phase 6 Documentation Spec

## DEFERRED DECISIONS

| Decision | Context | Target Resolution | Blocker |
|----------|---------|-------------------|---------|
| Quantization Required? | OSV-005 - Is 4-bit quantization required for RT submission? | Post-Phase 3 | Clarify RT requirements |
| Monitoring Dashboard? | OSV-006 - W&B dashboard required or just logging? | Post-Phase 5 | Time constraints |
| CommandRisk Integration? | Merge CommandRisk detection rules into CLI-Tuner guardrails? | Post-RT submission | CLI-Tuner must complete first |
| Testing Strategy | TDD or test-after? Coverage threshold? | Phase 0 | Architect spec needed |

---

## HOW TO USE THIS INDEX

### For User (Michael)
1. **Check Status:** See which phases are ready, which are pending
2. **Reference Issues:** Use OSV-XXX IDs in conversations (e.g., "Did we address OSV-003?")
3. **Track Progress:** Update "Success Gate" checkboxes as phases complete
4. **Self-Service:** Find spec locations without asking Architect

### For Architect
1. **Create New Specs:** Update this index when writing phase specs
2. **Mark Issues Addressed:** Link issue IDs to spec locations (line numbers)
3. **Update Status:** Move specs through maturity levels (L0 → L1 → L2 → L3)
4. **Cross-Reference:** Ensure phase specs link back to this index

### For Overseer
1. **Validate Claims:** Check that specs address claimed issues
2. **Review Systematically:** Validate specs at locations specified
3. **Approve Transitions:** Update status from L2 → L3 when approved
4. **Track Handoffs:** Verify artifacts match "Artifacts Produced" lists

### For Primary Engineer
1. **Find Implementation Specs:** Look up phase spec location
2. **Check Dependencies:** Verify previous phases complete before starting
3. **Deliver Artifacts:** Ensure outputs match "Artifacts Produced" lists
4. **Update Status:** Mark phases L4 when code complete

---

## VERSION HISTORY

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-15 | Initial creation, Phase 0/1 specs ready | Architect |
| 1.1 | 2026-01-15 | Phase 0/1 approved by Overseer, marked L3 (PE-Ready) | Overseer |
| 1.2 | 2026-01-15 | Phase 2 spec linked, artifacts prepared, priorities updated | PE |

---

**Next Update Trigger:** After Phase 2 smoke test results are reviewed

**Maintained By:** Architect (design phase), Overseer (validation phase), PE (implementation phase)

