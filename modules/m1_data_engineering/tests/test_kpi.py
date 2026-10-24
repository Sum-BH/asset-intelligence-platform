from datetime import datetime

import polars as pl
from aip_data.kpi import machine_kpis


def test_kpi_math():
    ts = lambda h: datetime(2026, 1, 1, h)  # noqa: E731
    events = pl.DataFrame(
        {
            "machine_id": ["M1"] * 5,
            "event": ["run", "failure", "repair_done", "failure", "repair_done"],
            "timestamp": [ts(0), ts(4), ts(6), ts(14), ts(15)],
        }
    )
    kpi = machine_kpis(events).row(0, named=True)
    # span 15h, downtime (2h + 1h) = 3h, 2 failures -> MTBF = 6h, MTTR = 1.5h
    assert kpi["n_failures"] == 2
    assert abs(kpi["mtbf_h"] - 6.0) < 1e-6
    assert abs(kpi["mttr_h"] - 1.5) < 1e-6
    assert abs(kpi["availability"] - 6.0 / 7.5) < 1e-6
