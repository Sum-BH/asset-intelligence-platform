# %% [markdown]
# # NB28 — One-Command Deploy + 15-Minute Quickstart (R37, R38)
# Generates the full-stack compose file (all profiles), the API Dockerfile, the
# quickstart doc, and NUMBERS.md — the consolidated measured figures that the business
# case must trace to. Verified: compose YAML parses with every expected service, the
# quickstart's asserted commands all exist in the Makefile, NUMBERS.md is populated
# from real artifacts. Target: `aip_serve/deploy.py` + deploy/.

# %%
import json
from pathlib import Path

import yaml

REPO = Path("../..")
DATA = REPO / "data"

# %% [markdown]
# ## 1. Full-stack compose (extends the Phase-0 baseline with the API)

# %%
compose = yaml.safe_load((REPO / "deploy/docker/docker-compose.yml").read_text())
compose["services"]["api"] = {
    "build": {"context": "../..", "dockerfile": "deploy/docker/Dockerfile.api"},
    "container_name": "aip-api",
    "environment": ["AIP_API_KEY=${AIP_API_KEY:-local-dev-key}",
                    "AIP_MLFLOW_TRACKING_URI=http://mlflow:5000",
                    "AIP_QDRANT_URL=http://qdrant:6333"],
    "ports": ["8000:8000"],
    "volumes": ["../../data:/app/data"],
    "depends_on": ["mlflow"],
    "restart": "unless-stopped",
}
# activate every profile service by default in the full-stack file
for svc in ("qdrant", "redpanda"):
    compose["services"][svc].pop("profiles", None)
full = REPO / "deploy/docker/docker-compose.full.yml"
full.write_text(yaml.safe_dump(compose, sort_keys=False))

parsed = yaml.safe_load(full.read_text())
expected = {"postgres", "mlflow", "qdrant", "redpanda", "api"}
assert set(parsed["services"]) == expected, parsed["services"].keys()
assert "8000:8000" in parsed["services"]["api"]["ports"]
assert all("profiles" not in s for s in parsed["services"].values())
print("compose.full:", sorted(parsed["services"]))

(REPO / "deploy/docker/Dockerfile.api").write_text('''\
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
COPY shared ./shared
COPY modules ./modules
RUN uv sync --no-dev
COPY . .
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "aip_serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
''')

# %% [markdown]
# ## 2. NUMBERS.md — every measured figure, traced to its artifact

# %%
def read(name):
    p = DATA / "processed" / name
    return json.loads(p.read_text()) if p.exists() else {}

rows = []
best = read("nb03_best.json")
if best:
    rows.append(("RUL best baseline", f"{best['best_model']} RMSE {best['rmse']:.2f} cycles",
                 "nb03_best.json (synthetic FD00S; re-measure on real FD001)"))
ret = read("retrieval_metrics.json")
if ret:
    rows.append(("RAG retrieval", f"hit@5 {ret['hybrid']['hit@5']}, MRR {ret['hybrid']['mrr']}",
                 "retrieval_metrics.json"))
fai = read("faithfulness_scores.json")
if fai:
    rows.append(("Report faithfulness", f"mean {fai['mean']}", "faithfulness_scores.json"))
cal = read("calibration_report.json")
if cal:
    rows.append(("Visual anomaly detection", f"mean image-AUROC {cal['mean_auroc']}",
                 "calibration_report.json"))
d15, d16, d17 = read("nb15_detections_meta.json"), read("nb16_hotspots.json"), read("nb17_fod.json")
if d15:
    rows.append(("Defect detection", f"mAP@0.5 {d15['map50']} (classical baseline)",
                 "nb15_detections_meta.json"))
if d16:
    rows.append(("Thermal hotspots", f"recall@IoU0.5 {d16['recall_iou50']}", "nb16_hotspots.json"))
if d17:
    rows.append(("FOD detection", f"mAP@0.5 {d17['map50']}", "nb17_fod.json"))
ev = read("nb23_eval_baseline.json").get("summary", {})
if ev:
    rows.append(("Agent eval", f"{ev['n_scenarios']} scenarios, groundedness "
                 f"{ev['groundedness']}, {ev['declines']} declines", "nb23_eval_baseline.json"))
sch = read("nb21_schedule.json")
if sch:
    rows.append(("Scheduler scale", f"50 WOs placed in {sch['scale']['seconds']}s "
                 f"({sch['scale']['status']}, optimality proven: "
                 f"{sch['scale']['optimality_proven']})", "nb21_schedule.json"))
demo = read("nb24_demo_audit.json")
if demo:
    rows.append(("End-to-end demo", f"{len(demo['alarms'])} alarms → "
                 f"{len(demo['schedule'])} scheduled in {demo['elapsed_seconds']}s",
                 "nb24_demo_audit.json"))
dr = read("nb27_drift.json")
if dr:
    rows.append(("Drift monitoring", f"latency {dr['detection_latency_windows']} windows, "
                 f"stationary FP rate {dr['false_positive_rate_stationary']}", "nb27_drift.json"))

md = ["# NUMBERS.md — measured platform figures", "",
      "Every business-case number must trace to a line here. Figures marked *synthetic*",
      "are sandbox measurements proving the code path; re-measure on real data before",
      "publishing externally.", "",
      "| Metric | Value | Source artifact |", "|---|---|---|"]
md += [f"| {a} | {b} | `{c}` |" for a, b, c in rows]
(REPO / "docs/business/NUMBERS.md").write_text("\n".join(md) + "\n")
print(f"NUMBERS.md: {len(rows)} measured figures")
assert len(rows) >= 8, "NUMBERS.md too sparse — run earlier phase notebooks first"

# %% [markdown]
# ## 3. Quickstart — commands must actually exist

# %%
quick = '''# 15-Minute Quickstart

## Prerequisites
Docker, [uv](https://docs.astral.sh/uv/), and (for local LLM features) Ollama.

## 1. Install and verify (3 min)
```bash
make setup          # create venv, install the uv workspace
make test           # module tests
make smoke          # environment readiness check
```

## 2. Start services (2 min)
```bash
make up             # MLflow (:5000) + Postgres
make up-full        # everything: + Qdrant, Redpanda, API (:8000)
curl localhost:8000/health
```

## 3. Run a phase notebook chain (10 min)
```bash
make data                       # download public datasets
make nb-exec PHASE=phase1       # ingestion -> features -> models -> diagnosis
make nb-exec PHASE=phase4       # tools -> agent -> optimizer -> MCP -> demo
```

## 4. See the loop close
```bash
make demo           # NB24: alarms -> drafts -> approval -> optimal schedule
```

All datasets are public; no proprietary data is used anywhere in this project.
'''
(REPO / "docs/QUICKSTART.md").write_text(quick)

makefile = (REPO / "Makefile").read_text()
needed = ["setup:", "test:", "smoke:", "up:", "up-full:", "data:", "nb-exec:", "demo:"]
missing = [t for t in needed if t not in makefile]
if missing:
    extra = "\nup-full:          ## Start the full stack (all services + API)\n" \
            "\tdocker compose -f deploy/docker/docker-compose.full.yml up -d\n" \
            "\ndemo:             ## Run the end-to-end loop demo (NB24)\n" \
            "\tcd notebooks/phase4 && uv run jupytext --to ipynb --set-kernel python3 " \
            "--execute nb24_end_to_end_demo.py\n"
    (REPO / "Makefile").write_text(makefile + extra)
    makefile = (REPO / "Makefile").read_text()
still = [t for t in needed if t not in makefile]
print("quickstart targets present:", [t.rstrip(':') for t in needed])
assert not still, f"quickstart references missing targets: {still}"

json.dump({"services": sorted(parsed["services"]), "numbers_rows": len(rows),
           "quickstart_targets": needed},
          open(DATA / "processed/nb28_deploy.json", "w"), indent=1)
print("NB28 complete.")
