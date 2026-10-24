# %% [markdown]
# # NB12 — SPIKE: Time-Series Foundation Models vs Phase 1 Baselines (R18)
# Builds the frozen evaluation harness (identical slice: per-unit last cycle on the
# same features/labels as NB03), registers baselines, and benchmarks any forecaster
# behind one interface. Foundation models (Chronos / TimesFM / Moirai) self-skip
# without their packages; the harness + comparison machinery runs everywhere.

# %%
import importlib.util
import json
from pathlib import Path
import numpy as np
import polars as pl

DATA = Path("../../data")
FD = next((DATA / "processed").glob("features_*_test.parquet")).stem.split("_")[1]
test = pl.read_parquet(DATA / f"processed/features_{FD}_test.parquet")
CONTEXT = 48  # cycles of history given to each forecaster

def nasa_score(y, p):
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.exp(-d[d < 0] / 13) - 1) + np.sum(np.exp(d[d >= 0] / 10) - 1))

# per-unit context series (sensor s2 trajectory) + true last-cycle RUL
units = []
for _, u in test.sort(["unit_id", "cycle"]).group_by("unit_id"):
    s = u["s2"].to_numpy()
    units.append({"unit_id": int(u["unit_id"][0]),
                  "context": s[-CONTEXT:] if len(s) >= CONTEXT else s,
                  "rul_true": int(u["rul"][-1])})
y_true = np.array([u["rul_true"] for u in units])
print(f"eval slice: {len(units)} units, context={CONTEXT} cycles, fd={FD}")

# %% [markdown]
# ## The harness: any forecaster = f(context_series) → rul_estimate

# %%
RESULTS = {}
def benchmark(name, fn):
    pred = np.clip([fn(u["context"]) for u in units], 0, None)
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    RESULTS[name] = {"rmse": round(rmse, 2), "nasa": round(nasa_score(y_true, pred), 1)}
    print(f"{name:28s} RMSE={rmse:6.2f}  NASA={RESULTS[name]['nasa']:8.1f}")

# baselines that run everywhere
benchmark("naive-constant-62.5", lambda c: 62.5)
def drift_extrapolate(c):
    # linear-trend time-to-threshold on the (scaled) degradation proxy
    if len(c) < 8: return 62.5
    t = np.arange(len(c)); k, b = np.polyfit(t, c, 1)
    if abs(k) < 1e-6: return 100.0
    return float(np.clip((1.0 - c[-1]) / k if k > 0 else (c[-1]) / -k, 0, 125))
benchmark("linear-drift-extrapolation", drift_extrapolate)

# Phase 1 tracked baselines, read from MLflow for the comparison table
import mlflow
mlflow.set_tracking_uri(f"sqlite:///{(DATA / 'processed/mlflow_local.db').resolve()}")
runs = mlflow.search_runs(experiment_names=["aip/m2/rul-cmapss"])
for _, r in runs.iterrows():
    fam, rmse = r.get("tags.model_family"), r.get("metrics.rmse")
    if fam and rmse == rmse:
        RESULTS.setdefault(f"phase1/{fam}",
                           {"rmse": round(float(rmse), 2),
                            "nasa": round(float(r.get("metrics.nasa_score", float("nan"))), 1)})

# %% [markdown]
# ## Foundation models (self-skip without packages; run on workstation)

# %%
for name, module, note in [("chronos", "chronos", "amazon chronos-bolt-small"),
                           ("timesfm", "timesfm", "google timesfm-1.0-200m"),
                           ("moirai", "uni2ts", "salesforce moirai-small")]:
    if importlib.util.find_spec(module) is None:
        print(f"SKIP {name}: `pip install {module}` on workstation ({note})")
        continue
    # workstation path sketch: forecast s2 forward, time-to-threshold → RUL proxy
    # implement per-model predict() here when running with the package present.

table = sorted(RESULTS.items(), key=lambda kv: kv[1]["rmse"])
json.dump({"fd_set": FD, "context": CONTEXT, "results": dict(table)},
          open(DATA / "processed/nb12_spike_results.json", "w"), indent=2)
print("\ncomparison table (asc RMSE):")
for k, v in table:
    print(f"  {k:30s} {v['rmse']:>7.2f}  {v['nasa']:>10.1f}")
assert any(k.startswith("phase1/") for k, _ in table), "Phase 1 baselines must appear in the table"
print("NB12 complete.")
