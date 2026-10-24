# Asset Intelligence Platform — Claude Code Agent Brief

> Read this file first, every session. It is the single source of truth for how to
> work in this repo. Do not modify it without explicit instruction.

---

## What this project is

An open-source, on-premise, multimodal industrial asset intelligence stack that closes
the full industrial decision loop:

**sense → detect → diagnose → explain → prescribe → schedule → act → learn**

All on affordable hardware, with a human approving every decision. No cloud lock.
No proprietary data. Every claim cites evidence.

One-liner: *"From raw sensor noise to an approved, optimally scheduled work order —
automatically, explainably, entirely inside the plant's four walls."*

---

## Repo layout

```
asset-intelligence-platform/
├── pyproject.toml              # uv workspace root
├── Makefile                    # setup / test / lint / nb-exec / demo
├── CHANGELOG.md
├── shared/aip_core/            # config, paths, logging, dataset registry
├── modules/
│   ├── m1_data_engineering/    # aip-data
│   ├── m2_signal_intelligence/ # aip-signal
│   ├── m3_vision_inspection/   # aip-vision
│   ├── m4_genai_rag/           # aip-rag
│   ├── m5_agentic_orchestration/ # aip-agents
│   └── m6_mlops_deployment/    # aip-serve
├── notebooks/
│   ├── phase1/   NB01–NB06  (signal intelligence)
│   ├── phase2/   NB07–NB12  (GenAI + RAG)
│   ├── phase3/   NB13–NB18  (computer vision)
│   ├── phase4/   NB19–NB24  (agents + optimizer)
│   └── phase5/   NB25–NB28  (MLOps + deploy)
├── data/           # git-ignored; never commit data files
├── docs/
│   ├── adr/        # Architecture Decision Records 001–003
│   ├── business/   # business-case.md, NUMBERS.md
│   └── governance/ # governance-note.md, model-cards/
└── deploy/docker/  # docker-compose.yml + .full.yml + Dockerfile.api
```

---

## Environment — read every word

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python env | `C:\vscode_hywork\dsml_projects\mlenv` |
| Project root | `C:\vscode_hywork\dsml_projects\assetintelligenceplatform` |
| Package manager | `uv` — all commands prefixed `uv run` |
| Shell | PowerShell (or Git Bash) |
| Path separator | Use `pathlib.Path` in all code — never hardcode `C:\` |

All code must use `aip_core.paths` for data/processed/interim paths.
Never hardcode Windows paths in Python.

---

## Hardware constraints — LAPTOP FIRST

### Laptop (current machine — execute Phase 1 and Phase 2 here)
- GPU: **RTX 3050 4GB VRAM** — very tight
- RAM: 16GB DDR5
- Storage: 512GB SSD

#### What this means in practice
- **Ollama models: 3B class only** — `llama3.2:3b` or `qwen2.5:3b`
  Never attempt a 7B or 14B pull on this machine — it will OOM
- **PyTorch (NB04 CNN):** CPU-only. Do not enable CUDA. Skip CUDA assertions.
- **No Qdrant server** on laptop — use local-file mode (`QdrantClient(path=...)`)
- **No anomalib, ultralytics (YOLO), VLM inference** on laptop
- **Phases 1 and 2 run fine** — DuckDB, sklearn, BM25, MLflow SQLite, report composer
- **Phase 3 and beyond:** workstation only (RTX 4060 Ti 16GB)

### Workstation (target after laptop phases are green)
- GPU: RTX 4060 Ti 16GB VRAM
- RAM: 64GB DDR5
- Full capability: all models, all phases

---

## Phase execution order — follow this strictly

Do NOT start a later phase until the earlier phase's notebook chain is green.

| Phase | Notebooks | Where to run | Gate |
|---|---|---|---|
| **1** | NB01–NB06 | Laptop | `make nb-exec PHASE=phase1` → zero errors |
| **2** | NB07–NB12 | Laptop | `make nb-exec PHASE=phase2` → zero errors |
| **3** | NB13–NB18 | Workstation | Requires RTX 4060 Ti |
| **4** | NB19–NB24 | Workstation | Requires RTX 4060 Ti + Phase 1–3 artifacts |
| **5** | NB25–NB28 | Workstation | Requires all prior phases |

---

## Phase 1 — what each notebook does

| NB | Week | Builds | Key output |
|---|---|---|---|
| NB01 | 3 | C-MAPSS ingestion + EDA | DuckDB `cycles`/`engines` tables |
| NB02 | 4 | Feature engineering | `features_{FD}_{split}.parquet` |
| NB03 | 5 | RUL baselines + MLflow | `nb03_best.json` + MLflow runs |
| NB04 | 6 | 1D-CNN | CPU-only on laptop; self-skips without CUDA |
| NB05 | 7 | Bearing health index + KPIs | Health-index plot + MTBF/MTTR |
| NB06 | 8 | LLM diagnostic summary | Dry-run prompt if Ollama absent |

Execution order: NB01 → NB02 → NB03 → NB05 → NB06 → NB04 (NB04 last, CPU-only)

---

## Phase 2 — what each notebook does

| NB | Builds | Key output |
|---|---|---|
| NB07 | Corpus validation | `corpus_manifest.json` |
| NB08 | Chunking + Qdrant index (local-file) | `qdrant_local/` + `chunking_report.json` |
| NB09 | Hybrid retrieval + eval | `retrieval_metrics.json` + `eval_set.jsonl` |
| NB10 | Diagnostic report generator | `reports/report_*.json` |
| NB11 | Faithfulness evaluation | `faithfulness_scores.json` |
| NB12 | TS foundation model spike | Comparison table (skips without package) |

---

## Key commands

```powershell
uv sync                              # install workspace (run once after clone)
uv run pytest -q                     # full test suite
uv run ruff check . --fix            # lint + auto-fix
make nb-exec PHASE=phase1            # execute Phase 1 notebook chain headlessly
make nb-exec PHASE=phase2            # execute Phase 2 notebook chain headlessly
make up                              # start MLflow + Postgres (needs Docker Desktop)
make data                            # download datasets (optional; synthetic works)
```

If `make` isn't available on Windows, run the nb-exec target manually:
```powershell
cd notebooks/phase1
for %f in (nb0*.py) do uv run jupytext --to ipynb --set-kernel python3 --execute %f
```

---

## Standing rules — never break these

1. **Nothing is done** without: green notebook chain + updated README section + note in NUMBERS.md
2. **Spikes after core.** NB04 (CNN), NB12 (TS foundation models) run AFTER NB01–NB03 / NB07–NB11 respectively
3. **No proprietary data.** Public datasets, synthetic data, self-authored knowledge only
4. **All tests must pass** before any commit: `uv run pytest -q` → green
5. **Ruff must be clean** before any commit: `uv run ruff check .` → "All checks passed!"
6. **Paths via aip_core.paths** — never hardcode `C:\` paths in code
7. **Ollama: 3B max on laptop.** `llama3.2:3b` or `qwen2.5:3b` only
8. **MLflow on laptop:** SQLite fallback — `sqlite:///data/processed/mlflow_local.db`
9. **Scope guardrail:** work only on the current phase. Do not add future-phase deps

---

## MLflow on laptop (no Docker required)

NB03 needs MLflow. Use the SQLite fallback — no Docker, no server:

```python
mlflow.set_tracking_uri("sqlite:///data/processed/mlflow_local.db")
```

This is already the default in the notebook when the server is unreachable.
If Docker Desktop is installed, `make up` gives you the full server on `:5000`.

---

## Jupyter kernel setup (run once)

```powershell
uv run python -m ipykernel install --user --name python3
```

If jupytext reports "no kernel matches":
```powershell
# Find the kernel json and check the argv path
jupyter kernelspec list
# Edit the kernel.json so argv[0] points to .venv/Scripts/python.exe (Windows)
```

---

## Data

- `data/` is git-ignored — never commit data files
- Synthetic C-MAPSS: generated automatically by `aip_data.synthetic` — no download needed
- Real NASA C-MAPSS: `make data` or `uv run python -m aip_data.download cmapss`
- Corpus documents: `data/corpus/*.md` — 12 starter docs included, rewrite in your own words

---

## What NOT to do on laptop

- Do NOT install torch with CUDA (`pip install torch --index-url .../cu121`) — 4GB VRAM will OOM
- Do NOT add `anomalib`, `ultralytics`, `langgraph`, `qwen2.5-vl` to deps yet
- Do NOT run `make up` unless Docker Desktop is installed and running
- Do NOT try to pull `llama3.1:8b` or `qwen2.5:14b` into Ollama here
- Do NOT run Phase 3, 4, or 5 notebooks on laptop
- Do NOT modify any file outside the current phase scope

---

## When something breaks — debug order

1. Check `uv sync` ran successfully
2. Check `uv run pytest -q` for which specific test fails
3. Check `uv run ruff check .` for lint errors
4. Check the notebook's last output cell for the actual error
5. Check `aip_core.paths` resolves correctly: `uv run python -c "from aip_core import paths; print(paths.root)"`
6. If MLflow fails: add `sqlite:///` fallback URI
7. If kernel fails: re-run `uv run python -m ipykernel install --user --name python3`

---

## Transfer to workstation (after laptop phases are green)

```powershell
# On laptop — commit and push
git add -A
git commit -m "Phase 1+2 green on laptop — ready for workstation"
git push

# On workstation
git clone <repo-url>
cd asset-intelligence-platform
uv sync
make smoke      # should print READY FOR PHASE 3
make data       # downloads real NASA datasets
make nb-exec PHASE=phase1   # re-run with real data + CUDA
```

Real-data workstation metrics go into NUMBERS.md — those are the numbers for Write-up #1.
