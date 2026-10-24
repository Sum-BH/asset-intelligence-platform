"""NASA C-MAPSS ingestion: parse -> DuckDB -> RUL labels.

C-MAPSS text format: space-separated, no header, 26 columns:
    unit_id, cycle, op_setting_1..3, sensor_1..21

Public functions:
    parse_cmapss_file(path)          -> polars DataFrame
    add_rul_labels(df, cap=125)      -> DataFrame with 'rul' column (train sets)
    attach_test_rul(df, rul_path)    -> per-unit final RUL for test sets
    load_to_duckdb(df, fd_set, con)  -> writes engines + cycles tables
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
from aip_core.logging import get_logger

log = get_logger("aip_data.cmapss")

OP_COLS = [f"op{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle", *OP_COLS, *SENSOR_COLS]


def parse_cmapss_file(path: str | Path) -> pl.DataFrame:
    """Parse one C-MAPSS train/test text file into a typed DataFrame."""
    path = Path(path)
    df = pl.read_csv(
        path,
        separator=" ",
        has_header=False,
        truncate_ragged_lines=True,
        new_columns=COLUMNS,
        schema_overrides={"unit_id": pl.Int32, "cycle": pl.Int32},
    )
    # NASA files end lines with trailing spaces -> possible null padding columns
    df = df.select(COLUMNS).drop_nulls(subset=["unit_id", "cycle"])
    log.info("parsed %s: %d rows, %d units", path.name, df.height, df["unit_id"].n_unique())
    return df


def add_rul_labels(df: pl.DataFrame, cap: int = 125) -> pl.DataFrame:
    """Piecewise-linear RUL: max_cycle(unit) - cycle, clipped at `cap`.

    Capping is standard practice on C-MAPSS: early-life degradation is
    unobservable, so an uncapped linear target teaches the model noise.
    """
    return df.with_columns(
        (pl.col("cycle").max().over("unit_id") - pl.col("cycle")).clip(upper_bound=cap).alias("rul")
    )


def attach_test_rul(df: pl.DataFrame, rul_path: str | Path, cap: int = 125) -> pl.DataFrame:
    """For test sets: RUL_*.txt gives remaining life AFTER each unit's last row."""
    final_rul = pl.read_csv(rul_path, has_header=False, new_columns=["final_rul"]).with_row_index(
        "idx"
    )
    final_rul = final_rul.with_columns((pl.col("idx") + 1).cast(pl.Int32).alias("unit_id"))
    df = df.join(final_rul.select("unit_id", "final_rul"), on="unit_id", how="left")
    return df.with_columns(
        (pl.col("cycle").max().over("unit_id") - pl.col("cycle") + pl.col("final_rul"))
        .clip(upper_bound=cap)
        .alias("rul")
    ).drop("final_rul")


def load_to_duckdb(
    df: pl.DataFrame, fd_set: str, con: duckdb.DuckDBPyConnection, split: str = "train"
) -> None:
    """Idempotent load into engines + cycles tables."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS engines (
            fd_set VARCHAR, split VARCHAR, unit_id INTEGER, max_cycle INTEGER,
            PRIMARY KEY (fd_set, split, unit_id)
        )
    """)
    sensor_ddl = ", ".join(f"{c} DOUBLE" for c in [*OP_COLS, *SENSOR_COLS])
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS cycles (
            fd_set VARCHAR, split VARCHAR, unit_id INTEGER, cycle INTEGER,
            {sensor_ddl}, rul INTEGER
        )
    """)
    con.execute("DELETE FROM cycles  WHERE fd_set = ? AND split = ?", [fd_set, split])
    con.execute("DELETE FROM engines WHERE fd_set = ? AND split = ?", [fd_set, split])

    if "rul" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias("rul"))
    tbl = df.with_columns(pl.lit(fd_set).alias("fd_set"), pl.lit(split).alias("split"))
    con.register("_incoming", tbl.to_arrow())
    con.execute(f"""
        INSERT INTO cycles
        SELECT fd_set, split, unit_id, cycle, {", ".join([*OP_COLS, *SENSOR_COLS])}, rul
        FROM _incoming
    """)
    con.execute(
        """
        INSERT INTO engines
        SELECT fd_set, split, unit_id, MAX(cycle)
        FROM cycles WHERE fd_set = ? AND split = ?
        GROUP BY fd_set, split, unit_id
    """,
        [fd_set, split],
    )
    con.unregister("_incoming")
    n = con.execute(
        "SELECT COUNT(*) FROM cycles WHERE fd_set=? AND split=?", [fd_set, split]
    ).fetchone()[0]
    log.info("loaded %s/%s: %d cycle rows", fd_set, split, n)
