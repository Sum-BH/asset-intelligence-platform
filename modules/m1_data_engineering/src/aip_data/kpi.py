"""Reliability & availability KPIs from a machine-event log.

Input contract: DataFrame with columns
    machine_id (str), event (str: 'run'|'failure'|'repair_done'),
    timestamp (datetime)

KPIs:
    mtbf   - mean time between failures (uptime hours / #failures)
    mttr   - mean time to repair (failure -> repair_done)
    availability - MTBF / (MTBF + MTTR)   [availability component of OEE]
"""

from __future__ import annotations

import polars as pl


def machine_kpis(events: pl.DataFrame) -> pl.DataFrame:
    ev = events.sort(["machine_id", "timestamp"])

    fails = ev.filter(pl.col("event") == "failure").sort("timestamp")
    repairs = ev.filter(pl.col("event") == "repair_done").sort("timestamp")

    # MTTR: pair each failure with the next repair_done on the same machine
    paired = fails.join_asof(
        repairs.rename({"timestamp": "repair_ts"}).select("machine_id", "repair_ts"),
        left_on="timestamp",
        right_on="repair_ts",
        by="machine_id",
        strategy="forward",
    ).with_columns(
        ((pl.col("repair_ts") - pl.col("timestamp")).dt.total_minutes() / 60.0).alias(
            "repair_hours"
        )
    )
    mttr = paired.group_by("machine_id").agg(pl.col("repair_hours").mean().alias("mttr_h"))

    # MTBF: observed span minus total downtime, over failure count
    span = ev.group_by("machine_id").agg(
        ((pl.col("timestamp").max() - pl.col("timestamp").min()).dt.total_minutes() / 60.0).alias(
            "span_h"
        )
    )
    downtime = paired.group_by("machine_id").agg(
        pl.col("repair_hours").sum().alias("down_h"),
        pl.len().alias("n_failures"),
    )
    kpi = (
        span.join(downtime, on="machine_id", how="left")
        .join(mttr, on="machine_id", how="left")
        .with_columns(
            pl.col("n_failures").fill_null(0),
            pl.col("down_h").fill_null(0.0),
        )
        .with_columns(
            pl.when(pl.col("n_failures") > 0)
            .then((pl.col("span_h") - pl.col("down_h")) / pl.col("n_failures"))
            .otherwise(None)
            .alias("mtbf_h")
        )
        .with_columns(
            (pl.col("mtbf_h") / (pl.col("mtbf_h") + pl.col("mttr_h"))).alias("availability")
        )
        .select("machine_id", "n_failures", "mtbf_h", "mttr_h", "availability")
        .sort("machine_id")
    )
    return kpi
