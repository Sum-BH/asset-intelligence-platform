"""Thermal hotspot detection (promoted from NB16).

Detector and ground truth both speak degC: extent at >1 degC above background,
components kept when their peak exceeds the seed threshold.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label


def detect_hotspots(frame: np.ndarray, seed_dt: float = 3.0, extent_dt: float = 1.0,
                    min_area: int = 9, bg_sigma: float = 20) -> list[dict]:
    resid = frame - gaussian_filter(frame, bg_sigma)
    lab, n = label(resid > extent_dt)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        peak = float(resid[ys, xs].max())
        if len(xs) < min_area or peak < seed_dt:
            continue
        out.append({"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "peak_dt": round(peak, 1)})
    return out


def severity_band(delta_t: float) -> str:
    """Thermal inspection convention (see corpus heur-thermal-0008)."""
    if delta_t > 40:
        return "immediate"
    if delta_t >= 15:
        return "repair_soon"
    if delta_t >= 5:
        return "plan_repair"
    return "monitor"
