# Changelog

## v1.0.0 — 2026-07-21

First complete release: the full industrial decision loop —
sense → detect → diagnose → explain → prescribe → schedule → act → learn —
running on-premise on a single desktop GPU workstation, with a human approving
every decision.

### Modules
- **aip-core** — config, paths, logging, dataset registry
- **aip-data** — C-MAPSS ingestion, synthetic generator, DuckDB model, reliability KPIs
- **aip-signal** — features, RUL baselines (RMSE + NASA score), bearing health index,
  LLM diagnostics, 1D-CNN
- **aip-rag** — corpus validation, chunking, BM25/hybrid retrieval, cited reports,
  grounding metrics
- **aip-vision** — PatchCore-lite anomaly detection, defect/FOD detection with mAP,
  thermal hotspots, VisionFinding schema
- **aip-agents** — three typed tools, LangGraph HITL work-order agent, CP-SAT scheduler,
  MCP server, evaluation metrics
- **aip-serve** — FastAPI service layer, registry promotion gates, drift monitoring

### Guarantees enforced in code
- No path reaches an approved work order without a human decision (graph interrupt)
- Every raised action cites a real knowledge-base chunk; thin coverage → refusal
- Registry promotion refuses NASA-score regressions; Production needs a written reason
- API approvals require an approver identity (403 otherwise)

### Development history
- 28 notebooks across 6 phases, jupytext-paired (ADR-002), all executing with zero errors
- Notebook logic promoted to modules in Phase 6 with reconciliation recorded (ADR-003)
- 47+ tests ported from in-notebook asserts

### Known limitations
- Sandbox metrics come from synthetic data; re-measure on real datasets before external use
- Dense retrieval, anomalib, YOLO, VLM and torch paths are workstation extensions
- CP-SAT at 50 work orders returns a feasible (not proven-optimal) schedule within 10 s
