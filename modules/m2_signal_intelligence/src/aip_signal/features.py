"""Feature engineering for RUL modelling on C-MAPSS-style cycle data.

Pipeline (fit on train only, apply to both splits):
    1. select_informative_sensors  - drop near-constant sensors by variance
    2. build_features              - rolling mean/std + first differences
    3. fit_scaler / apply_scaler   - min-max per feature, train-fitted

Design choice: verify sensor informativeness with a variance check instead of
hard-coding the folklore FD001 drop-list; the same code then works on FD002+
and on synthetic data.
"""

from __future__ import annotations

import polars as pl

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
DEFAULT_WINDOWS = (5, 10, 20)


def select_informative_sensors(train: pl.DataFrame, rel_std_threshold: float = 1e-4) -> list[str]:
    """Keep sensors whose std/|mean| exceeds the threshold (unit-free)."""
    stats = train.select(
        [pl.col(c).std().alias(f"{c}_std") for c in SENSOR_COLS]
        + [pl.col(c).mean().abs().alias(f"{c}_mean") for c in SENSOR_COLS]
    ).row(0, named=True)
    keep = [
        c
        for c in SENSOR_COLS
        if stats[f"{c}_std"] is not None
        and stats[f"{c}_std"] / max(stats[f"{c}_mean"], 1e-9) > rel_std_threshold
    ]
    return keep


def build_features(
    df: pl.DataFrame, sensors: list[str], windows: tuple[int, ...] = DEFAULT_WINDOWS
) -> pl.DataFrame:
    """Per-unit rolling statistics and first differences for selected sensors."""
    exprs: list[pl.Expr] = []
    for c in sensors:
        exprs.append(pl.col(c).diff().over("unit_id").alias(f"{c}_d1"))
        for w in windows:
            exprs.append(
                pl.col(c).rolling_mean(w, min_samples=1).over("unit_id").alias(f"{c}_rm{w}")
            )
            exprs.append(
                pl.col(c).rolling_std(w, min_samples=2).over("unit_id").alias(f"{c}_rs{w}")
            )
    out = df.sort(["unit_id", "cycle"]).with_columns(exprs)
    feat_cols = feature_columns(sensors, windows)
    return out.with_columns([pl.col(c).fill_null(0.0) for c in feat_cols])


def feature_columns(sensors: list[str], windows: tuple[int, ...] = DEFAULT_WINDOWS) -> list[str]:
    cols: list[str] = []
    for c in sensors:
        cols += [c, f"{c}_d1"]
        for w in windows:
            cols += [f"{c}_rm{w}", f"{c}_rs{w}"]
    return cols


def fit_scaler(train: pl.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    row = train.select(
        [pl.col(c).min().alias(f"{c}__min") for c in cols]
        + [pl.col(c).max().alias(f"{c}__max") for c in cols]
    ).row(0, named=True)
    return {c: (row[f"{c}__min"], row[f"{c}__max"]) for c in cols}


def apply_scaler(df: pl.DataFrame, scaler: dict[str, tuple[float, float]]) -> pl.DataFrame:
    exprs = []
    for c, (lo, hi) in scaler.items():
        rng = (hi - lo) or 1.0
        exprs.append(((pl.col(c) - lo) / rng).alias(c))
    return df.with_columns(exprs)
