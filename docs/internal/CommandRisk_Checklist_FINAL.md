# CommandRisk Implementation Checklist
## Based on Northstar v3.0 — Security Detection Engine

**Version:** 1.0  
**Date:** 2025-01-15  
**Purpose:** Track implementation progress against Northstar requirements

---

## Progress Summary

**Total Items:** 85  
**Completed:** 0/85 ☐  
**Target Timeline:** 4-5 weeks (solo, part-time)

### By Phase:
- Phase 1 (Rules): 0/35 ☐
- Phase 2 (Heuristics): 0/20 ☐
- Phase 3 (ML Classifier): 0/15 ☐ (Optional)
- Phase 4 (Integration): 0/8 ☐ (Optional)
- Operational: 0/7 ☐

---

## PHASE 1: RULE-BASED DETECTION (Weeks 1-2)

### Rule Development

**Critical Rules (Always Block):**
- [ ] Rule: `destructive_rm` - Recursive root deletion (rm -rf /)
- [ ] Rule: `fork_bomb` - Fork bomb pattern (:(){ :|:& };:)
- [ ] Rule: `disk_wipe` - Disk wipe (dd if=/dev/zero of=/dev/sda)
- [ ] Rule: `blanket_permissions` - Blanket permissions (chmod -R 777 /)
- [ ] Rule: `remote_execution_curl` - Blind remote execution (curl ... | bash)
- [ ] Rule: `remote_execution_wget` - Blind remote execution (wget ... | sh)
- [ ] Rule: `format_disk` - Format disk (mkfs.* /dev/sd*)
- [ ] Rule: `direct_disk_write` - Direct disk write (> /dev/sd*)

**High-Risk Rules (Flag for Review):**
- [ ] Rule: `recursive_deletion` - Recursive deletion (rm -rf <directory>)
- [ ] Rule: `world_writable` - World-writable permissions (chmod 777)
- [ ] Rule: `sudo_command` - Sudo usage (sudo <command>)
- [ ] Rule: `netcat_listen` - Listening netcat (nc -l <port>)
- [ ] Rule: `base64_decode` - Base64 encoded commands
- [ ] Rule: `eval_remote` - Eval with remote fetch (eval $(curl ...))
- [ ] Rule: `data_exfiltration` - Data exfiltration (tar ... | nc)
- [ ] Rule: `password_extraction` - Password file access (cat /etc/shadow)

**Obfuscation Detection:**
- [ ] Rule: `excessive_whitespace` - Unusual whitespace patterns
- [ ] Rule: `excessive_quoting` - Excessive quoting ("")
- [ ] Rule: `hex_encoding` - Hex-encoded commands
- [ ] Rule: `unicode_tricks` - Unicode homoglyphs

### MITRE ATT&CK Mapping

- [ ] Map all rules to MITRE tactics (TA####)
- [ ] Map all rules to MITRE techniques (T####)
- [ ] Create MITRE coverage matrix (tactics vs rules)
- [ ] Document coverage gaps (uncovered tactics)

### Rule Testing

**Test Set Creation:**
- [ ] Create benign test set (5,000 common commands)
- [ ] Create malicious test set (500 dangerous commands)
- [ ] Create evasion test set (100 obfuscated variants)

**Test Execution:**
- [ ] Run all rules on benign test set
- [ ] Calculate FPR (target: ≈0%)
- [ ] Run all rules on malicious test set
- [ ] Calculate TPR (target: ≥90%)
- [ ] Document false positives (for rule refinement)
- [ ] Document false negatives (for new rules)

### Rule Engine Implementation

- [ ] Define rule schema (YAML format)
- [ ] Implement rule parser (load YAML → rule objects)
- [ ] Implement pattern matcher (regex-based)
- [ ] Implement rule executor (run all rules on command)
- [ ] Implement result aggregator (combine rule matches)
- [ ] Unit test rule engine (positive and negative cases)

### CLI Tool

- [ ] Create `commandrisk` CLI script
- [ ] Implement: `commandrisk analyze "<command>"`
- [ ] Implement: `commandrisk analyze --file <log_file>`
- [ ] Implement: `commandrisk test` (run on test sets)
- [ ] Implement: `commandrisk rules` (list all rules)
- [ ] Color-coded output (green=safe, yellow=review, red=dangerous)

### Output Formats

- [ ] Implement JSONL output formatter
- [ ] Implement CLI output formatter (human-readable)
- [ ] Test output formats (validate schema)

**Phase 1 Success Gate:** Critical FPR ≈ 0%, Critical TPR ≥ 90%

---

## PHASE 2: HEURISTIC SCORING (Weeks 2-3)

### Scoring Rubric Development

**Base Scores:**
- [ ] Define base scores for command features (file deletion, network, sudo, etc.)
- [ ] Assign points: file_deletion=+3, recursive=-2, force=+2, root=+10, etc.
- [ ] Document scoring rationale (why each feature has this score)

**Context Multipliers:**
- [ ] Define user context multipliers (root=×1.5, service_account=×1.2)
- [ ] Define host context multipliers (production=×2.0, critical=×3.0)
- [ ] Define time context multipliers (off-hours=×1.5, weekend=×1.3)
- [ ] Define directory context multipliers (/tmp=×1.2, /etc=×1.5)

**Risk Thresholds:**
- [ ] Set initial thresholds (0-29=safe, 30-59=review, 60-84=dangerous, 85-100=critical)
- [ ] Create config file: `configs/scoring_config.yaml`

### Scoring Engine Implementation

- [ ] Implement feature extractor (parse command → features)
- [ ] Implement base score calculator (features → base_score)
- [ ] Implement context multiplier calculator (context → multipliers)
- [ ] Implement final score calculator (base × multipliers → final_score)
- [ ] Implement risk level mapper (final_score → risk_level)
- [ ] Unit test scoring engine (known inputs → expected scores)

### Context Integration

- [ ] Implement user context parser (extract username, groups)
- [ ] Implement host context parser (extract hostname, criticality)
- [ ] Implement time context parser (extract timestamp, calculate business_hours)
- [ ] Implement directory context parser (extract cwd, classify directory)
- [ ] Mock context for testing (no real CMDB/identity provider needed)

### Threshold Tuning

- [ ] Run scoring engine on validation set (1,000 commands)
- [ ] Plot score distribution (histogram)
- [ ] Analyze false positives (benign commands with high scores)
- [ ] Adjust base scores (reduce FP sources)
- [ ] Analyze false negatives (dangerous commands with low scores)
- [ ] Adjust base scores (increase sensitivity)
- [ ] Re-test on test set (validate tuning)

### Output Enhancement

- [ ] Add risk_score to JSONL output
- [ ] Add context_factors to JSONL output
- [ ] Add score_breakdown to CLI output (show how score was calculated)

**Phase 2 Success Gate:** Suspicious/Malicious detection ≥ 80% without FP spike

---

## PHASE 3: SEMANTIC CLASSIFIER (Week 4 - Optional)

### Embedding Model Selection

- [ ] Research sentence-transformers models (MPNet, MiniLM, etc.)
- [ ] Select model (balance: size vs accuracy vs speed)
- [ ] Download and test model (verify inference works)

### Training Data Preparation

**Malicious Set:**
- [ ] Collect 500 dangerous commands (base set)
- [ ] Generate obfuscated variants (whitespace, encoding, aliases)
- [ ] Total: 500 × 20 variants = 10,000 malicious examples

**Benign Set:**
- [ ] Use existing benign test set (5,000 commands)
- [ ] Generate common command variants (increase diversity)
- [ ] Total: 10,000 benign examples

**Dataset Balancing:**
- [ ] Balance classes (50% malicious, 50% benign)
- [ ] Split: 80% train, 10% val, 10% test

### Classifier Training/Fine-Tuning

- [ ] Embed all training examples (command → 768-dim vector)
- [ ] Train similarity threshold (cosine similarity > 0.85 → malicious)
- [ ] OR: Fine-tune classifier (binary classification head)
- [ ] Validate on validation set (tune threshold)

### Evasion Test Set

- [ ] Create evasion test set (100 obfuscated dangerous commands)
- [ ] Categories: whitespace, encoding, unicode, aliases
- [ ] Run classifier on evasion set
- [ ] Calculate TPR (target: ≥80%)

### Integration with Rules + Heuristics

- [ ] Add semantic layer to detection pipeline
- [ ] If rules match: use rules (highest confidence)
- [ ] If no rules match: use heuristic scoring
- [ ] If score ambiguous: use semantic classifier
- [ ] Combine results (rules + heuristics + semantic)

**Phase 3 Success Gate:** Evasion set TPR ≥ 80%, benign FPR ≈ 0%

---

## PHASE 4: CLI-TUNER INTEGRATION (Week 5 - Optional)

### Post-Generation Validation

- [ ] Create integration script (`integration/cli_tuner_validator.py`)
- [ ] Load CLI-Tuner model
- [ ] Generate command from instruction
- [ ] Pass command to CommandRisk
- [ ] If dangerous: block and return error
- [ ] If safe: return command
- [ ] Test integration (100 instructions)

### Integration Testing

- [ ] Test: Benign instructions → safe commands
- [ ] Test: Suspicious instructions → dangerous commands blocked
- [ ] Test: Edge cases (ambiguous instructions)
- [ ] Measure: False positive rate (benign blocked)
- [ ] Measure: False negative rate (dangerous passed)

### Feedback Loop Integration

- [ ] Log all CLI-Tuner outputs with CommandRisk assessment
- [ ] Track: instruction → command → risk_level
- [ ] Identify: High-risk patterns (frequently dangerous)
- [ ] Action: Add to CLI-Tuner guardrails (pre-generation filtering)

### Documentation

- [ ] Create integration guide (`docs/integration.md`)
- [ ] Document: Setup, usage, configuration
- [ ] Provide: Code examples (Python, CLI)
- [ ] Document: Known limitations, future enhancements

**Phase 4 Success Gate:** Dangerous CLI-Tuner outputs blocked before user sees

---

## OPERATIONAL MATURITY (Ongoing)

### Rule Versioning

- [ ] Initialize Git repository for rules
- [ ] Create CHANGELOG.md (document all rule changes)
- [ ] Tag releases (v1.0.0, v1.1.0, etc.)
- [ ] Document rule lifecycle (draft → review → test → deploy)

### Detection Metrics

- [ ] Implement metrics calculator (TPR, FPR, Precision, F1)
- [ ] Run weekly evaluation on test sets
- [ ] Log metrics to `evaluation/metrics_YYYY-MM-DD.json`
- [ ] Plot trends (weekly TPR/FPR over time)
- [ ] Alert if degradation detected (>5% drop)

### Feedback Loop

- [ ] Implement false positive reporting (CLI flag: --report-fp)
- [ ] Implement false negative reporting (CLI flag: --report-fn)
- [ ] Create feedback database (`monitoring/feedback.jsonl`)
- [ ] Schedule weekly review (30 minutes)
- [ ] Action: Update rules based on feedback

**Operational Success Gate:** All monitoring, versioning, and improvement processes operational

---

## QUALITY THRESHOLDS

### Detection Quality
- ✅ Critical TPR ≥ 90% (dangerous commands detected)
- ✅ Critical FPR ≈ 0% (benign commands passed)
- ✅ Evasion TPR ≥ 80% (obfuscated variants detected)

### Explainability
- ✅ Every detection has rule ID + description
- ✅ Every detection has MITRE ATT&CK mapping
- ✅ Risk scores are consistent and justifiable

### Operational Quality
- ✅ Rules versioned in Git (full change history)
- ✅ Metrics tracked weekly (TPR, FPR, Precision, F1)
- ✅ Feedback loop operational (FP/FN → improvements)

---

## OUTPUT DELIVERABLES

### Code Artifacts
- [ ] Rule engine (`engine/rules.py`)
- [ ] Scoring engine (`engine/scoring.py`)
- [ ] Semantic classifier (`engine/semantic.py` - optional)
- [ ] CLI tool (`cli.py`)
- [ ] Test suites (`tests/`)

### Rule Artifacts
- [ ] Rule definitions (YAML files in `rules/`)
- [ ] MITRE ATT&CK mappings (`docs/MITRE_MAPPING.md`)
- [ ] Rule documentation (`docs/RULES.md`)
- [ ] CHANGELOG.md (version history)

### Evaluation Artifacts
- [ ] Test sets (benign, malicious, evasion)
- [ ] Metrics logs (`evaluation/metrics/`)
- [ ] Evaluation report (`docs/EVALUATION.md`)

### Documentation
- [ ] Northstar (this document)
- [ ] README.md (project overview)
- [ ] ARCHITECTURE.md (system design)
- [ ] Integration guide (if Phase 4 complete)

---

## COMPLETION TRACKING

**Current Phase:** Not Started  
**Days Elapsed:** 0  
**Items Completed Today:** 0

**Update this section daily to track progress.**

---

## NOTES

### Rule Development Tips
- Start with high-confidence critical rules (low FP risk)
- Test each rule on 10+ benign variants before deploying
- Document false positives immediately (for tuning)
- Peer review all rules (2+ security engineers if possible)

### Scoring Tuning Tips
- Start with conservative base scores (low FP)
- Increase scores gradually (improve TPR)
- Use validation set for tuning (not test set)
- Context multipliers should be evidence-based (not arbitrary)

### Testing Strategy
- Test incrementally (don't wait until end)
- Regression tests after every rule change
- Evasion tests should be diverse (whitespace, encoding, unicode, aliases)

---

**Last Updated:** 2025-01-15  
**Next Review:** After Phase 1 completion
