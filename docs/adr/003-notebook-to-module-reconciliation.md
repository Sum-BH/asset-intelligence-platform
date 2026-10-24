# ADR-003: Notebook→module reconciliation decisions (Phase 6)

- **Status:** Accepted
- **Date:** 2026-07-21
- **Deciders:** Sumedh Bhalerao

## Context
Phase 6 promotes notebook logic (NB01–NB28) into the module skeleton. Where a
verified reference module already existed (Phases 0–1), and where notebook code
diverged during Phases 2–5, a choice had to be recorded rather than made silently.

## Decisions

1. **`WorkOrder` name collision (the one that actually broke a run).**
   NB20 defined a business `WorkOrder` (evidence, citations, approver, status);
   NB21 defined a solver-input `WorkOrder` (duration, window, skill). Executing both
   in NB24 raised `TypeError: unexpected keyword argument 'wo_id'`.
   → `aip_agents.scheduler.WorkOrderSpec` (solver input) vs
   `aip_agents.graph.WorkOrder` (business object). A module-level alias
   `WorkOrder = WorkOrderSpec` remains in scheduler.py for migrating notebook code.

2. **Phase-1 reference modules kept; notebooks were the divergent copy.**
   `aip_data.cmapss`, `aip_signal.features/models/health_index/diagnose` were verified
   first and remain authoritative. NB01–NB06 duplicated that logic for authoring
   convenience; the modules win.

3. **Retrieval: BM25 promoted as the default path.**
   NB08/NB09 used Qdrant + embeddings; NB10 and all Phase-4 tools used BM25 for
   dependency-light determinism. `BM25Retriever` is the promoted default;
   dense/hybrid retrieval stays a workstation extension. Rationale: the agent tools
   must work in CI and on the laptop without a running vector database.

4. **Detector parameters frozen at their measured values.**
   `aip_vision.defects.detect` keeps the stripe-removal + hysteresis + peak-relative
   extent design found in NB15 (mAP@0.5 0.63 on the hard synthetic set). Test gates
   are set slightly below measured values (0.55 / 0.60 / 0.85) so ordinary numeric
   drift does not produce false CI failures while real regressions still fail.

5. **Tool data access: `fetchall`, not arrow.**
   `query_signals` originally used `polars.from_arrow`, which raised inside the
   LangGraph runtime on zero-row results. Plain `fetchall` is promoted.

6. **Paths come from `aip_core.paths`, never notebook-relative literals.**
   All promoted functions accept explicit path overrides for testability.

## Consequences
+ One implementation per behaviour; notebooks become documentation.
+ Tests encode the notebook asserts, so gates survive without executing notebooks.
- Notebooks still contain their inline copies (they are archived, not rewritten);
  readers must know the modules are authoritative. Stated in README.
