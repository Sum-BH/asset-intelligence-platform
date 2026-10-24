# %% [markdown]
# # NB02 — Week 4: Feature Engineering
# **Objective:** informative-sensor selection (variance-based, NOT the folklore list),
# per-unit rolling features, train-only scaling, persist Parquet.
#
# **Contracts produced (§7.2):** `features_{FD}_{split}.parquet`; feature column ordering.
# Conversion target: `aip_signal/features.py`.

# %%
from pathlib import Path

import duckdb
import polars as pl

DATA_DIR = Path("../../data")
con = duckdb.connect(str(DATA_DIR / "interim/nb_aip.duckdb"), read_only=True)
FD = con.execute("SELECT DISTINCT fd_set FROM cycles").fetchone()[0]
train = pl.from_arrow(con.execute(
    "SELECT * FROM cycles WHERE fd_set=? AND split='train'", [FD]).arrow())
test = pl.from_arrow(con.execute(
    "SELECT * FROM cycles WHERE fd_set=? AND split='test'", [FD]).arrow())
print(FD, train.shape, test.shape)

# %% [markdown]
# ## 1. Sensor selection — relative std (unit-free), fit on TRAIN only

# %%
S = [f"s{i}" for i in range(1, 22)]
stats = train.select([pl.col(c).std().alias(f"{c}_std") for c in S]
                     + [pl.col(c).mean().abs().alias(f"{c}_mean") for c in S]).row(0, named=True)
SENSORS = [c for c in S if (stats[f"{c}_std"] or 0) / max(stats[f"{c}_mean"], 1e-9) > 1e-4]
print(f"kept {len(SENSORS)}/21 sensors:", SENSORS)

# %% [markdown]
# ## 2. Rolling features + first differences (per unit), null-safe

# %%
WINDOWS = (5, 10, 20)


def feature_columns(sensors):
    cols = []
    for c in sensors:
        cols += [c, f"{c}_d1"] + [f"{c}_r{k}{w}" for w in WINDOWS for k in ("m", "s")]
    return cols


def build_features(df):
    exprs = []
    for c in SENSORS:
        exprs.append(pl.col(c).diff().over("unit_id").alias(f"{c}_d1"))
        for w in WINDOWS:
            exprs.append(pl.col(c).rolling_mean(w, min_samples=1)
                         .over("unit_id").alias(f"{c}_rm{w}"))
            exprs.append(pl.col(c).rolling_std(w, min_samples=2)
                         .over("unit_id").alias(f"{c}_rs{w}"))
    out = df.sort(["unit_id", "cycle"]).with_columns(exprs)
    return out.with_columns([pl.col(c).fill_null(0.0) for c in feature_columns(SENSORS)])


FEATS = feature_columns(SENSORS)
train_f, test_f = build_features(train), build_features(test)
assert train_f.select(FEATS).null_count().sum_horizontal().item() == 0

# %% [markdown]
# ## 3. Min-max scaling — fitted on TRAIN only, applied to both

# %%
row = train_f.select([pl.col(c).min().alias(f"{c}__lo") for c in FEATS]
                     + [pl.col(c).max().alias(f"{c}__hi") for c in FEATS]).row(0, named=True)
scaler = {c: (row[f"{c}__lo"], row[f"{c}__hi"]) for c in FEATS}


def apply_scaler(df):
    return df.with_columns([((pl.col(c) - lo) / ((hi - lo) or 1.0)).alias(c)
                            for c, (lo, hi) in scaler.items()])


train_f, test_f = apply_scaler(train_f), apply_scaler(test_f)
mins, maxs = train_f.select(FEATS).min().row(0), train_f.select(FEATS).max().row(0)
assert min(mins) >= -1e-9 and max(maxs) <= 1 + 1e-9

# %% [markdown]
# ## 4. Persist — contract filenames

# %%
out = DATA_DIR / "processed"
out.mkdir(parents=True, exist_ok=True)
train_f.write_parquet(out / f"features_{FD}_train.parquet")
test_f.write_parquet(out / f"features_{FD}_test.parquet")
import json
(out / f"feature_meta_{FD}.json").write_text(json.dumps(
    {"sensors": SENSORS, "features": FEATS, "windows": WINDOWS,
     "feature_version": "v1"}, indent=2))
print("NB02 complete →", out)
