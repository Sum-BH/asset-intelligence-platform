"""RUL baseline models (Random Forest, Gradient Boosting) + MLflow tracking.

Evaluation:
    rmse        - standard root-mean-square error in cycles
    nasa_score  - official C-MAPSS asymmetric score: late predictions
                  (pred > true, d > 0) are penalised harder than early ones.
                  score = sum( exp(-d/13) - 1 ) for d < 0
                        + sum( exp( d/10) - 1 ) for d >= 0,  d = pred - true
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from aip_core import settings
from aip_core.logging import get_logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

log = get_logger("aip_signal.models")


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = d < 0
    return float(np.sum(np.exp(-d[early] / 13.0) - 1.0) + np.sum(np.exp(d[~early] / 10.0) - 1.0))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


@dataclass
class TrainResult:
    model_name: str
    rmse: float
    nasa: float
    run_id: str | None = None


def last_cycle_matrix(df: pl.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Test-style evaluation: one prediction per unit at its last observed cycle."""
    last = df.filter(pl.col("cycle") == pl.col("cycle").max().over("unit_id"))
    return last.select(feat_cols).to_numpy(), last["rul"].to_numpy()


def train_baselines(
    train: pl.DataFrame,
    test: pl.DataFrame,
    feat_cols: list[str],
    fd_set: str,
    feature_version: str = "v1",
    use_mlflow: bool = True,
    tracking_uri: str | None = None,
) -> list[TrainResult]:
    x_tr = train.select(feat_cols).to_numpy()
    y_tr = train["rul"].to_numpy()
    x_te, y_te = last_cycle_matrix(test, feat_cols)

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }

    mlflow = None
    if use_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
        mlflow.set_experiment("aip/m2/rul-cmapss")

    results: list[TrainResult] = []
    for name, model in models.items():
        model.fit(x_tr, y_tr)
        pred = np.clip(model.predict(x_te), 0, None)
        res = TrainResult(name, rmse(y_te, pred), nasa_score(y_te, pred))

        if mlflow:
            with mlflow.start_run(run_name=f"{name}-{fd_set}") as run:
                mlflow.set_tags(
                    {
                        "dataset": "cmapss",
                        "fd_set": fd_set,
                        "model_family": name,
                        "feature_version": feature_version,
                    }
                )
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({"rmse": res.rmse, "nasa_score": res.nasa})
                mlflow.log_dict({"features": feat_cols}, "features.json")
                res.run_id = run.info.run_id

        log.info("%s [%s]: RMSE=%.2f  NASA=%.1f", name, fd_set, res.rmse, res.nasa)
        results.append(res)
    return results
