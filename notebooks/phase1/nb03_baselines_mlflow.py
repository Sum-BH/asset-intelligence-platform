# %% [markdown]
# # NB03 — Week 5: RUL Baselines + NASA Score + MLflow
# **Objective:** RF & GBM on last-cycle features; RMSE + asymmetric NASA score;
# runs tracked in MLflow under `aip/m2/rul-cmapss` with the frozen tag set.
#
# Conversion target: `aip_signal/models.py`.

# %%
import json
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

DATA_DIR = Path("../../data")
meta = json.loads(next((DATA_DIR / "processed").glob("feature_meta_*.json")).read_text())
FD = next((DATA_DIR / "processed").glob("features_*_train.parquet")).stem.split("_")[1]
FEATS = meta["features"]
train = pl.read_parquet(DATA_DIR / f"processed/features_{FD}_train.parquet")
test = pl.read_parquet(DATA_DIR / f"processed/features_{FD}_test.parquet")

# %% [markdown]
# ## 1. NASA score — late predictions penalised harder than early

# %%
def nasa_score(y, p):
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.exp(-d[d < 0] / 13) - 1) + np.sum(np.exp(d[d >= 0] / 10) - 1))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


assert nasa_score([100], [110]) > nasa_score([100], [90]) > 0  # asymmetry check

# %% [markdown]
# ## 2. Train/eval — one prediction per test unit at its last observed cycle

# %%
x_tr, y_tr = train.select(FEATS).to_numpy(), train["rul"].to_numpy()
last = test.filter(pl.col("cycle") == pl.col("cycle").max().over("unit_id"))
x_te, y_te = last.select(FEATS).to_numpy(), last["rul"].to_numpy()

mlflow.set_tracking_uri(f"sqlite:///{(DATA_DIR / 'processed/mlflow_local.db').resolve()}")
mlflow.set_experiment("aip/m2/rul-cmapss")

MODELS = {
    "random_forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=5,
                                           n_jobs=-1, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=4, random_state=42),
}
results = {}
for name, model in MODELS.items():
    model.fit(x_tr, y_tr)
    pred = np.clip(model.predict(x_te), 0, None)
    m = {"rmse": rmse(y_te, pred), "nasa_score": nasa_score(y_te, pred)}
    with mlflow.start_run(run_name=f"{name}-{FD}"):
        mlflow.set_tags({"dataset": "cmapss", "fd_set": FD, "model_family": name,
                         "feature_version": meta["feature_version"]})
        mlflow.log_metrics(m)
        mlflow.log_dict({"features": FEATS}, "features.json")
    results[name] = m
    print(f"{name:20s} RMSE={m['rmse']:6.2f}  NASA={m['nasa_score']:8.1f}")

best = min(results, key=lambda k: results[k]["rmse"])
json.dump({"best_model": best, **results[best]},
          open(DATA_DIR / "processed/nb03_best.json", "w"), indent=2)
print("best:", best, "| NB03 complete.")
