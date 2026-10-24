# %% [markdown]
# # NB04 — Week 6: 1D-CNN Sequence Model (WORKSTATION)
# Requires torch + CUDA (RTX 4060 Ti). Cells self-skip if torch is absent so the
# notebook suite always executes end-to-end. Conversion target: `aip_signal/cnn_rul.py`.

# %%
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

TORCH = importlib.util.find_spec("torch") is not None
DATA_DIR = Path("../../data")
print("torch available:", TORCH)

# %% [markdown]
# ## 1. Sliding windows per unit → (N, window, features), target = RUL at window end

# %%
meta = json.loads(next((DATA_DIR / "processed").glob("feature_meta_*.json")).read_text())
FD = next((DATA_DIR / "processed").glob("features_*_train.parquet")).stem.split("_")[1]
FEATS, WINDOW = meta["features"], 30
train = pl.read_parquet(DATA_DIR / f"processed/features_{FD}_train.parquet")
test = pl.read_parquet(DATA_DIR / f"processed/features_{FD}_test.parquet")


def make_windows(df, window=WINDOW):
    xs, ys = [], []
    for _, u in df.sort(["unit_id", "cycle"]).group_by("unit_id"):
        arr, rul = u.select(FEATS).to_numpy(), u["rul"].to_numpy()
        if len(arr) < window:
            arr = np.vstack([np.repeat(arr[:1], window - len(arr), axis=0), arr])
            rul = np.concatenate([np.repeat(rul[:1], window - len(rul)), rul])
        for end in range(window, len(arr) + 1):
            xs.append(arr[end - window:end]); ys.append(rul[end - 1])
    return np.stack(xs), np.array(ys, dtype=np.float32)


x_smoke, y_smoke = make_windows(train.filter(pl.col("unit_id") <= 2))
print("window smoke:", x_smoke.shape, y_smoke.shape)  # torch-free shape check

# %% [markdown]
# ## 2. Model + training (skips without torch)

# %%
if TORCH:
    import mlflow
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x_tr, y_tr = make_windows(train)
    lastw = test.filter(pl.col("cycle") >= pl.col("cycle").max().over("unit_id") - WINDOW + 1)
    x_te, y_te = make_windows(lastw)

    model = nn.Sequential(
        nn.Conv1d(len(FEATS), 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    lossf = nn.SmoothL1Loss()
    dl = DataLoader(TensorDataset(torch.tensor(x_tr).float().permute(0, 2, 1),
                                  torch.tensor(y_tr)), batch_size=256, shuffle=True)
    mlflow.set_tracking_uri(f"sqlite:///{(DATA_DIR / 'processed/mlflow_local.db').resolve()}")
    mlflow.set_experiment("aip/m2/rul-cmapss")
    with mlflow.start_run(run_name=f"cnn1d-{FD}"):
        mlflow.set_tags({"dataset": "cmapss", "fd_set": FD, "model_family": "cnn1d",
                         "feature_version": meta["feature_version"]})
        for ep in range(30):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(model(xb).squeeze(-1), yb)
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = np.clip(model(torch.tensor(x_te).float().permute(0, 2, 1).to(dev))
                           .squeeze(-1).cpu().numpy(), 0, None)
        r = float(np.sqrt(np.mean((y_te - pred) ** 2)))
        mlflow.log_metric("rmse", r)
        print("cnn1d RMSE:", r)
else:
    print("SKIPPED: install torch on the workstation (see runbook) and rerun.")
print("NB04 complete.")
