"""Synthetic C-MAPSS-format data generator (synthetic fault injector, v1).

Produces train/test/RUL text files byte-compatible with NASA's format so the
entire pipeline (parser -> DuckDB -> features -> models) is testable and
demoable without the real download. Degradation model per unit:

    sensor(t) = baseline + drift_direction * (t / life)^gamma * scale + noise

A subset of sensors is flat (constant + noise) to mimic the uninformative
sensors in FD001, which the feature selector must learn to drop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from aip_core.logging import get_logger

log = get_logger("aip_data.synthetic")

N_SENSORS = 21
FLAT_SENSORS = {0, 4, 5, 9, 15, 17, 18}  # 0-based -> s1,s5,s6,s10,s16,s18,s19


def _unit_trajectory(rng: np.random.Generator, life: int) -> np.ndarray:
    t = np.arange(1, life + 1, dtype=float)
    frac = t / life
    out = np.empty((life, N_SENSORS))
    for j in range(N_SENSORS):
        base = 100.0 + 10.0 * j
        if j in FLAT_SENSORS:
            out[:, j] = base + rng.normal(0, 0.01, life)
        else:
            direction = 1.0 if j % 2 == 0 else -1.0
            gamma = rng.uniform(1.2, 2.5)
            scale = rng.uniform(3.0, 8.0)
            noise = rng.normal(0, 0.15, life)
            out[:, j] = base + direction * scale * frac**gamma + noise
    return out


def generate(
    out_dir: str | Path,
    n_train_units: int = 40,
    n_test_units: int = 20,
    seed: int = 7,
    fd_name: str = "FD00S",
) -> dict[str, Path]:
    """Write train_/test_/RUL_ files in NASA C-MAPSS format. Returns paths."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_units(n_units: int, truncate: bool):
        rows, final_ruls = [], []
        for unit in range(1, n_units + 1):
            life = int(rng.integers(140, 320))
            traj = _unit_trajectory(rng, life)
            ops = rng.normal([0.0, 0.0, 100.0], [0.002, 0.0003, 0.0], (life, 3))
            last = life
            if truncate:
                last = int(rng.integers(int(life * 0.4), life - 5))
                final_ruls.append(life - last)
            for c in range(last):
                vals = [unit, c + 1, *ops[c], *traj[c]]
                rows.append(
                    " ".join(f"{v:d}" if i < 2 else f"{v:.4f}" for i, v in enumerate(vals)) + " "
                )  # trailing space, as in the NASA files
        return rows, final_ruls

    train_rows, _ = write_units(n_train_units, truncate=False)
    test_rows, ruls = write_units(n_test_units, truncate=True)

    paths = {
        "train": out_dir / f"train_{fd_name}.txt",
        "test": out_dir / f"test_{fd_name}.txt",
        "rul": out_dir / f"RUL_{fd_name}.txt",
    }
    paths["train"].write_text("\n".join(train_rows) + "\n")
    paths["test"].write_text("\n".join(test_rows) + "\n")
    paths["rul"].write_text("\n".join(str(r) for r in ruls) + "\n")
    log.info(
        "synthetic %s: %d train rows, %d test rows, %d units w/ RUL",
        fd_name,
        len(train_rows),
        len(test_rows),
        len(ruls),
    )
    return paths
