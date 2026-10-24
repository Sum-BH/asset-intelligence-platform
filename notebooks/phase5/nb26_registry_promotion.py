# %% [markdown]
# # NB26 — Model Registry Promotion with Metric Gates (R36)
# Promotion to Staging requires the candidate to be no worse than the incumbent on the
# **NASA score** (the asymmetric metric that actually matters for maintenance — late
# predictions are the expensive ones). Production is a manual step requiring a written
# reason. Every promotion event is logged for audit.
# Verified: a regressed candidate is REFUSED. Target: `aip_serve/promotion.py`.

# %%
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np

DATA = Path("../../data")
mlflow.set_tracking_uri(f"sqlite:///{(DATA / 'processed/mlflow_local.db').resolve()}")
EXPERIMENT = "aip/m2/rul-cmapss"
LEDGER = DATA / "processed/promotion_ledger.json"

# %% [markdown]
# ## 1. Read real Phase-1 runs as promotion candidates

# %%
runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
cands = []
for _, r in runs.iterrows():
    fam, rmse, nasa = r.get("tags.model_family"), r.get("metrics.rmse"), r.get("metrics.nasa_score")
    if fam and rmse == rmse:
        cands.append({"run_id": r["run_id"][:8], "model_family": fam,
                      "rmse": round(float(rmse), 3),
                      "nasa_score": round(float(nasa), 1) if nasa == nasa else None})
cands.sort(key=lambda c: c["nasa_score"] if c["nasa_score"] is not None else 1e9)
print("candidates (best NASA first):")
for c in cands:
    print(f"  {c['model_family']:20s} rmse={c['rmse']:7.3f}  nasa={c['nasa_score']}")
assert cands, "no MLflow runs found — execute Phase 1 NB03 first"

# %% [markdown]
# ## 2. Promotion gate

# %%
def load_ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"events": [], "stages": {}}


def promote(candidate: dict, stage: str, reason: str | None = None, actor: str = "sumedh"):
    """Staging: automatic gate on NASA score. Production: manual + reason required."""
    led = load_ledger()
    incumbent = led["stages"].get(stage)
    decision, why = "promoted", "no incumbent"
    if stage == "Production" and not reason:
        decision, why = "refused", "Production promotion requires a written reason"
    elif incumbent:
        if candidate["nasa_score"] is None:
            decision, why = "refused", "candidate missing nasa_score"
        elif candidate["nasa_score"] > incumbent["nasa_score"]:
            decision = "refused"
            why = (f"NASA score regression: {candidate['nasa_score']} > "
                   f"incumbent {incumbent['nasa_score']}")
        else:
            why = (f"NASA improved {incumbent['nasa_score']} → {candidate['nasa_score']}")
    event = {"ts": datetime.now(UTC).isoformat(), "actor": actor, "stage": stage,
             "candidate": candidate, "decision": decision, "reason": reason, "why": why}
    led["events"].append(event)
    if decision == "promoted":
        led["stages"][stage] = candidate
    LEDGER.write_text(json.dumps(led, indent=1))
    print(f"  [{decision.upper():8s}] {candidate['model_family']} → {stage}: {why}")
    return event

# %% [markdown]
# ## 3. Exercise: first promotion, regression refusal, manual Production

# %%
LEDGER.unlink(missing_ok=True)
best, worst = cands[0], cands[-1]

e1 = promote(best, "Staging")
assert e1["decision"] == "promoted"

e2 = promote(worst, "Staging")            # worse NASA score → must be refused
assert e2["decision"] == "refused" and "regression" in e2["why"], "gate failed to block"

e3 = promote(best, "Production")          # no reason → refused
assert e3["decision"] == "refused"

e4 = promote(best, "Production", reason="Validated on FD001 holdout; approved by DS lead.")
assert e4["decision"] == "promoted"

# a synthetic improvement must pass
better = {**best, "run_id": "cand9999", "model_family": "cnn1d",
          "nasa_score": round(best["nasa_score"] * 0.8, 1), "rmse": best["rmse"] * 0.9}
e5 = promote(better, "Staging")
assert e5["decision"] == "promoted"

led = load_ledger()
print(f"\nledger: {len(led['events'])} events, "
      f"{sum(e['decision']=='refused' for e in led['events'])} refusals")
print("current stages:", {k: v["model_family"] for k, v in led["stages"].items()})
assert led["stages"]["Staging"]["model_family"] == "cnn1d"
print("NB26 complete.")
