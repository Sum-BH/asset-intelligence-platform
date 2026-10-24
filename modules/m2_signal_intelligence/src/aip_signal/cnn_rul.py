"""1D-CNN for RUL on sliding sensor windows. WORKSTATION-ONLY (requires torch).

Not part of the sandbox-verified suite: run on the RTX 4060 Ti.
    uv add torch --optional gpu   (or pip install torch with CUDA build)
    uv run python -m aip_signal.cnn_rul --fd FD001

Architecture: 3x (Conv1d -> BN -> ReLU) -> GAP -> MLP head. ~100k params,
trains FD001 in minutes on the 4060 Ti. Window: 30 cycles x selected sensors.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def make_windows(
    df: pl.DataFrame, feat_cols: list[str], window: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows per unit -> (N, window, n_features), targets = RUL at window end."""
    xs, ys = [], []
    for _, unit in df.sort(["unit_id", "cycle"]).group_by("unit_id"):
        arr = unit.select(feat_cols).to_numpy()
        rul = unit["rul"].to_numpy()
        if len(arr) < window:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr = np.vstack([pad, arr])
            rul = np.concatenate([np.repeat(rul[:1], window - len(rul)), rul])
        for end in range(window, len(arr) + 1):
            xs.append(arr[end - window : end])
            ys.append(rul[end - 1])
    return np.stack(xs), np.array(ys, dtype=np.float32)


def build_model(n_features: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )


def train(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feat_cols: list[str],
    fd_set: str,
    window: int = 30,
    epochs: int = 30,
    batch: int = 256,
    lr: float = 1e-3,
):
    import mlflow
    import torch
    from aip_core import settings
    from torch.utils.data import DataLoader, TensorDataset

    from aip_signal.models import last_cycle_matrix, nasa_score, rmse  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_tr, y_tr = make_windows(train_df, feat_cols, window)
    x_te, y_te = make_windows(
        test_df.filter(pl.col("cycle") >= pl.col("cycle").max().over("unit_id") - window + 1),
        feat_cols,
        window,
    )

    # torch Conv1d wants (N, channels, length)
    tr = TensorDataset(torch.tensor(x_tr, dtype=torch.float32).permute(0, 2, 1), torch.tensor(y_tr))
    loader = DataLoader(tr, batch_size=batch, shuffle=True)

    model = build_model(len(feat_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("aip/m2/rul-cmapss")
    with mlflow.start_run(run_name=f"cnn1d-{fd_set}"):
        mlflow.set_tags(
            {
                "dataset": "cmapss",
                "fd_set": fd_set,
                "model_family": "cnn1d",
                "feature_version": "v1",
            }
        )
        mlflow.log_params(
            {
                "window": window,
                "epochs": epochs,
                "lr": lr,
                "batch": batch,
                "n_features": len(feat_cols),
            }
        )
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
            mlflow.log_metric("train_loss", float(loss), step=epoch)

        model.eval()
        with torch.no_grad():
            pred = (
                model(torch.tensor(x_te, dtype=torch.float32).permute(0, 2, 1).to(device))
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        pred = np.clip(pred, 0, None)
        m_rmse, m_nasa = rmse(y_te, pred), nasa_score(y_te, pred)
        mlflow.log_metrics({"rmse": m_rmse, "nasa_score": m_nasa})
        print(f"cnn1d [{fd_set}] RMSE={m_rmse:.2f} NASA={m_nasa:.1f}")
    return model
