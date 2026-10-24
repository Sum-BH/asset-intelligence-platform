# Asset Intelligence Platform

Open-source, on-premise, multimodal industrial asset intelligence stack that
closes the full decision loop: **sense → detect → diagnose → explain →
prescribe → schedule → act → learn** — on affordable hardware, with a human
approving every decision.

> From raw sensor noise to an approved, optimally scheduled work order —
> automatically, explainably, entirely inside the plant's four walls.

## Modules (monorepo)

| # | Package | Scope | Status |
|---|---------|-------|--------|
| 1 | `aip-data` | Ingestion, synthetic generator, DuckDB model, KPIs | v1.0 |
| 2 | `aip-signal` | Features, RUL models, health index, LLM diagnostics | v1.0 |
| 3 | `aip-vision` | Anomaly/defect/thermal/FOD detection, VisionFinding schema | v1.0 |
| 4 | `aip-rag` | Corpus, chunking, retrieval, cited diagnostic reports | v1.0 |
| 5 | `aip-agents` | Tools, LangGraph HITL agent, CP-SAT scheduler, MCP server | v1.0 |
| 6 | `aip-serve` | FastAPI, registry promotion, drift monitoring | v1.0 |

Shared code lives in `shared/aip_core` (config, logging, dataset registry).

## Quick start

```bash
# prerequisites: uv (https://docs.astral.sh/uv/), Docker
make setup     # create venv, install workspace
make up        # start MLflow (http://localhost:5000) + Postgres
make data      # download Phase-1 public datasets into data/raw/
make smoke     # verify environment end-to-end
make test      # run tests
make lab       # JupyterLab (notebook-first workflow, see ADR-002)
make nb-exec   # execute the current phase notebook chain headlessly
```

All datasets are public (NASA C-MAPSS, IMS, AI4I 2020, MVTec AD, NEU, FOD-A);
no proprietary data is used anywhere in this project.

## Docs

- Architecture decision records: `docs/adr/`
- Public write-ups: `docs/writeups/`
- Business case & ROI model: `docs/business/`

## What the platform guarantees

These are enforced in code, not documentation:

- **No autonomous approval.** The agent graph is compiled with an interrupt before the
  approval node; no path reaches an approved work order without a human decision.
- **No uncited claims.** Every raised action cites a knowledge-base chunk. When
  retrieval coverage is thin the system refuses rather than inventing guidance.
- **No silent regressions.** Registry promotion refuses candidates that worsen the
  NASA score; Production promotion requires a written reason. Both are logged.
- **No anonymous approvals.** API approval endpoints require an approver identity.

## Repository layout

- `modules/` — the authoritative implementation (see ADR-003)
- `notebooks/` — 28 executed notebooks, phase by phase: how the platform was built,
  kept as documentation. Where notebook and module logic differ, **modules win**.
- `docs/` — [ADRs](docs/adr/README.md), [quickstart](docs/QUICKSTART.md),
  [business case](docs/business/business-case.md),
  [governance note](docs/governance/governance-note.md),
  [measured figures](docs/business/NUMBERS.md)

## Development

```bash
make lab                      # JupyterLab (notebook-first workflow, ADR-002)
make nb-exec PHASE=phase4     # execute a phase's notebook chain headlessly
make test                     # module test suite
make up-full                  # full stack incl. API on :8000
make demo                     # the end-to-end loop demo
```

## Status

**v1.0** — see [CHANGELOG.md](CHANGELOG.md). Sandbox metrics come from synthetic data
and prove the code paths; re-measure on real datasets before external use.
