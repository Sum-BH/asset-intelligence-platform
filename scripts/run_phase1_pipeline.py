"""Phase 1 end-to-end pipeline: data -> DuckDB -> features -> models -> diagnosis.

    uv run python scripts/run_phase1_pipeline.py --dataset synthetic
    uv run python scripts/run_phase1_pipeline.py --dataset real --fd FD001

'synthetic' generates C-MAPSS-format data (no download needed); 'real' expects
`python -m aip_data.download cmapss` to have populated data/raw/cmapss/.
MLflow: uses AIP_MLFLOW_TRACKING_URI; falls back to a local file store if the
server is unreachable, so the pipeline never blocks on Docker.
"""

from __future__ import annotations

import argparse

import duckdb
import requests
from aip_core import paths, settings
from aip_core.logging import get_logger
from aip_data import cmapss, synthetic
from aip_signal import diagnose, features, models

log = get_logger("pipeline.phase1")


def resolve_tracking_uri() -> str:
    try:
        requests.get(settings.mlflow_tracking_uri + "/health", timeout=2)
        return settings.mlflow_tracking_uri
    except Exception:
        uri = f"sqlite:///{paths.processed / 'mlflow_local.db'}"
        log.warning("MLflow server unreachable; using local file store %s", uri)
        return uri


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--fd", default="FD001", help="C-MAPSS subset (real only)")
    ap.add_argument("--rul-cap", type=int, default=125)
    args = ap.parse_args()

    paths.ensure()
    if args.dataset == "synthetic":
        fd = "FD00S"
        files = synthetic.generate(paths.synthetic / "cmapss")
        train_path, test_path, rul_path = files["train"], files["test"], files["rul"]
    else:
        fd = args.fd
        base = paths.raw / "cmapss"
        train_path, test_path, rul_path = (
            base / f"train_{fd}.txt",
            base / f"test_{fd}.txt",
            base / f"RUL_{fd}.txt",
        )

    # 1. ingest
    train = cmapss.add_rul_labels(cmapss.parse_cmapss_file(train_path), cap=args.rul_cap)
    test = cmapss.attach_test_rul(cmapss.parse_cmapss_file(test_path), rul_path, cap=args.rul_cap)
    con = duckdb.connect(settings.duckdb_path)
    cmapss.load_to_duckdb(train, fd, con, split="train")
    cmapss.load_to_duckdb(test, fd, con, split="test")

    # 2. features (fit decisions on train only)
    sensors = features.select_informative_sensors(train)
    log.info("informative sensors (%d): %s", len(sensors), sensors)
    feat_cols = features.feature_columns(sensors)
    train_f = features.build_features(train, sensors)
    test_f = features.build_features(test, sensors)
    scaler = features.fit_scaler(train_f, feat_cols)
    train_f = features.apply_scaler(train_f, scaler)
    test_f = features.apply_scaler(test_f, scaler)
    train_f.write_parquet(paths.processed / f"features_{fd}_train.parquet")
    test_f.write_parquet(paths.processed / f"features_{fd}_test.parquet")

    # 3. models + MLflow
    results = models.train_baselines(
        train_f, test_f, feat_cols, fd_set=fd, tracking_uri=resolve_tracking_uri()
    )
    best = min(results, key=lambda r: r.rmse)

    # 4. diagnostic summary (dry-run prints the prompt if Ollama is absent)
    ev = diagnose.Evidence(
        asset_id=f"{fd}-unit-001",
        rul_estimate_cycles=round(best.rmse * 3, 1),  # illustrative single-asset value
        rul_model=best.model_name,
        rul_rmse_cycles=round(best.rmse, 2),
        health_index_current=0.0,
        health_index_trend="flat",
        top_features=feat_cols[:5],
        notes="End-to-end pipeline demo run.",
    )
    try:
        text = diagnose.diagnose(ev)
        print("\n--- LLM DIAGNOSTIC SUMMARY ---\n" + text)
    except Exception as e:  # noqa: BLE001
        log.warning("Ollama unavailable (%s); showing rendered prompt instead", e)
        print("\n--- RENDERED PROMPT (dry run) ---\n" + diagnose.diagnose(ev, dry_run=True))

    print(f"\nBest baseline: {best.model_name}  RMSE={best.rmse:.2f}  NASA={best.nasa:.1f}")
    print("Feature files:", paths.processed)


if __name__ == "__main__":
    main()
