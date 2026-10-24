# Governance Note — Asset Intelligence Platform

*How decisions are made, who is accountable, and what the system will not do.*

## 1. Decision rights (the core design commitment)

| Stage | Decided by | Rationale |
|---|---|---|
| Detection (anomaly, RUL, defect) | Statistical/ML models | Reproducible, measurable, versioned |
| Diagnosis narrative | LLM, constrained to retrieved evidence | Language is the LLM's job; judgement is not |
| Pass/fail on inspection | Detector scores, never the VLM | The vision-language model *describes*; it does not decide |
| Work-order approval | **Human, always** | Structurally enforced: the agent graph interrupts before the approval node; no code path reaches `approved` without a human decision |
| Scheduling | CP-SAT constraint solver | A provably optimal answer exists; an LLM guessing schedules would undermine explainability |

**Summary: models detect, the LLM drafts, the solver decides, the human approves.**

## 2. Refusal as a feature

The system declines rather than fabricates:

- Report generation refuses when knowledge-base coverage is below threshold
  ("insufficient knowledge-base coverage") instead of inventing guidance.
- The agent raises no work order when signal and vision evidence are both absent —
  5 of 11 evaluation scenarios exercise this decline path deliberately.

Refusal behaviour is measured, not assumed.

## 3. Evaluation as change control

No model or prompt change ships without the eval harness re-run:

- **Retrieval:** hit@5, MRR on an authored eval set
- **Reports:** citation coverage and grounding score
- **Agent:** expectation match, action groundedness, evidence-consistency, decline correctness
- **Promotion:** registry gate refuses candidates that regress the NASA score
  (asymmetric — late RUL predictions are penalised harder, because they are the
  expensive failure mode in maintenance)

Baselines are committed to the repo so regressions are visible in diffs.

## 4. Auditability

Every work order records: the evidence snapshot it was built from, the knowledge-base
chunk IDs cited, the approver identity, and any reviewer note on rejection. Promotion
events are appended to a ledger with actor, decision and reason. API approvals require
an explicit approver header — anonymous approval is rejected (HTTP 403).

## 5. Model cards

One card per trained model under `docs/governance/model-cards/`, covering data,
intended use, metrics, limitations and drift triggers. A model without a card does not
get promoted past Staging.

## 6. Data protection and IP

- All development data is public, synthetic or self-authored. No employer or client IP
  is used anywhere in this project.
- Deployment is on-premise by design: process data, defect imagery and the knowledge
  base never leave the plant.
- Client engagements: their data stays in their environment; the knowledge base built
  during a pilot belongs to them.

## 7. Regulatory posture (EU AI Act awareness)

Industrial predictive maintenance is not a prohibited practice and is generally not
high-risk under the Act on its own. Two areas warrant care:

- **Worker-management adjacency.** Scheduling assigns work to named technicians. The
  platform must not be repurposed into performance monitoring or evaluation of workers
  — that adjacency is where the high-risk classification actually lives.
- **Transparency.** Generated diagnostic text is machine-produced and labelled as such;
  citations let any reader verify claims against source documents.

The human-in-the-loop gate, audit trail and refusal behaviour are the mitigations, and
they are architectural rather than procedural — they cannot be skipped by a busy user.

*This note reflects a considered engineering posture, not legal advice; a compliance
review is warranted before any regulated deployment.*

## 8. Incident playbook (sketch)

1. **Detect** — drift alerts (feature PSI > 0.2, prediction KS p < 0.01) or a rejected
   work-order pattern
2. **Contain** — demote the model to Staging; the API keeps serving the previous version
3. **Diagnose** — replay the eval harness scenarios against the suspect version
4. **Correct** — retrain or adjust retrieval; re-run gates
5. **Record** — append to the promotion ledger with reason
