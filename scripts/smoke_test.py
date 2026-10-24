"""Phase 0 smoke test: verifies the environment is ready for Phase 1.

Checks: package imports, repo paths, dataset registry, GPU visibility
(optional), MLflow server reachability (optional).
Exit code 0 = ready. Warnings do not fail the run.
"""

from __future__ import annotations

import sys


def check(label: str, fn, required: bool = True) -> bool:
    try:
        detail = fn()
        print(f"  [OK]   {label}" + (f" - {detail}" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001
        tag = "FAIL" if required else "WARN"
        print(f"  [{tag}] {label} - {e}")
        return not required


def main() -> int:
    print("Asset Intelligence Platform - Phase 0 smoke test\n")
    ok = True

    def imports():
        import aip_signal  # noqa: F401
        import sklearn  # noqa: F401

        return "core scientific stack + workspace packages import"

    def repo():
        from aip_core import paths

        paths.ensure()
        return f"root={paths.root}"

    def registry():
        from aip_core import DATASETS

        return f"{len(DATASETS)} datasets registered"

    def duck():
        import duckdb

        assert duckdb.sql("SELECT 21*2 AS x").fetchone()[0] == 42
        return f"duckdb {duckdb.__version__} query OK"

    def gpu():
        import torch

        assert torch.cuda.is_available(), "CUDA not available"
        return torch.cuda.get_device_name(0)

    def mlflow_server():
        import requests
        from aip_core import settings

        r = requests.get(settings.mlflow_tracking_uri + "/health", timeout=3)
        r.raise_for_status()
        return f"MLflow healthy at {settings.mlflow_tracking_uri}"

    ok &= check("Python package imports", imports)
    ok &= check("Repo paths resolve & data dirs exist", repo)
    ok &= check("Dataset registry", registry)
    ok &= check("DuckDB in-process query", duck)
    ok &= check("CUDA GPU visible (optional here; required on workstation)", gpu, required=False)
    ok &= check("MLflow server reachable (run `make up` first)", mlflow_server, required=False)

    print("\nRESULT:", "READY FOR PHASE 1" if ok else "FIX FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
