# %% [markdown]
# # NB27 — Drift Monitoring: Feature PSI + Prediction Drift (R35)
# PSI per feature against the training reference (alert > 0.2) and a KS test on the
# prediction distribution (alert p < 0.01). Alerts land in a DuckDB table the API can
# serve. Verified both ways: injected drift is DETECTED, and stationary data does not
# raise false alarms. Target: `aip_serve/drift.py`.

# %%
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.stats import ks_2samp

DATA = Path("../../data")
rng = np.random.default_rng(5)

# %% [markdown]
# ## 1. PSI

# %%
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))

# %% [markdown]
# ## 2. Reference from real Phase-1 features

# %%
FD = next((DATA / "processed").glob("features_*_train.parquet")).stem.split("_")[1]
train = pl.read_parquet(DATA / f"processed/features_{FD}_train.parquet")
meta = json.loads((DATA / f"processed/feature_meta_{FD}.json").read_text())
FEATS = meta["features"][:8]
ref = {f: train[f].to_numpy() for f in FEATS}
print(f"reference: {FD}, {train.height} rows, monitoring {len(FEATS)} features")

def make_batch(drifted: bool):
    out = {}
    for f in FEATS:
        base = rng.choice(ref[f], size=400, replace=True)
        out[f] = base + (0.25 + 0.4 * rng.random()) if drifted else base
    return out

# %% [markdown]
# ## 3. Detect: stationary must stay silent, drifted must alert

# %%
PSI_ALERT, KS_ALERT = 0.2, 0.01

def check_batch(batch, preds_ref=None, preds_cur=None):
    alerts = []
    for f in FEATS:
        v = psi(ref[f], batch[f])
        if v > PSI_ALERT:
            alerts.append({"kind": "feature_psi", "feature": f, "value": round(v, 4)})
    if preds_ref is not None:
        p = ks_2samp(preds_ref, preds_cur).pvalue
        if p < KS_ALERT:
            alerts.append({"kind": "prediction_ks", "feature": "rul_pred",
                           "value": round(float(p), 6)})
    return alerts

stationary = [check_batch(make_batch(False)) for _ in range(10)]
fp_rate = sum(bool(a) for a in stationary) / len(stationary)
print(f"stationary batches: {sum(bool(a) for a in stationary)}/10 raised alerts "
      f"(false-positive rate {fp_rate:.2f})")
assert fp_rate <= 0.1, "too many false positives on stationary data"

drift_alerts = check_batch(make_batch(True))
print(f"drifted batch: {len(drift_alerts)} alerts, e.g. {drift_alerts[:2]}")
assert len(drift_alerts) >= len(FEATS) // 2, "injected drift not detected"

pr_ref = rng.normal(60, 12, 500)
pr_cur = rng.normal(35, 12, 500)                      # RUL predictions collapsing
pred_alerts = check_batch(make_batch(False), pr_ref, pr_cur)
assert any(a["kind"] == "prediction_ks" for a in pred_alerts), "prediction drift missed"
print("prediction drift detected:", [a for a in pred_alerts if a["kind"] == "prediction_ks"])

# %% [markdown]
# ## 4. Detection latency + alerts table served by the API

# %%
latency = None
for i in range(1, 6):                                  # ramping drift
    b = {f: rng.choice(ref[f], 400, replace=True) + 0.06 * i for f in FEATS}
    if check_batch(b):
        latency = i
        break
print(f"detection latency: {latency} windows of ramping drift")
assert latency is not None and latency <= 5

con = duckdb.connect(str(DATA / "processed/aip.duckdb"))
con.execute("""CREATE TABLE IF NOT EXISTS alerts (
    ts TIMESTAMP, kind VARCHAR, feature VARCHAR, value DOUBLE, batch VARCHAR)""")
con.execute("DELETE FROM alerts WHERE batch = 'nb27-demo'")
now = datetime.now(UTC).replace(tzinfo=None)
for a in drift_alerts + pred_alerts:
    con.execute("INSERT INTO alerts VALUES (?,?,?,?,?)",
                [now, a["kind"], a["feature"], a["value"], "nb27-demo"])
n = con.execute("SELECT COUNT(*) FROM alerts WHERE batch='nb27-demo'").fetchone()[0]
con.close()
print(f"{n} alerts written to DuckDB alerts table")

json.dump({"psi_alert": PSI_ALERT, "ks_alert": KS_ALERT,
           "false_positive_rate_stationary": fp_rate,
           "drift_alerts": len(drift_alerts), "detection_latency_windows": latency,
           "alerts_persisted": n},
          open(DATA / "processed/nb27_drift.json", "w"), indent=1)
print("NB27 complete.")
