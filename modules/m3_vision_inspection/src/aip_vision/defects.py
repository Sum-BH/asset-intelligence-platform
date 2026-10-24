"""Defect / FOD detection and detection metrics (promoted from NB15/NB17).

Three lessons encoded: remove periodic background structure (column median) before
residuals; use hysteresis (strong seed, weak grow) instead of one threshold; refit
boxes on pixels above a fraction of the component peak so extents are scale-invariant.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label, median_filter


def detect(img: np.ndarray, k_hi: float = 3.0, k_lo: float = 1.2, min_area: int = 12,
           dil: int = 3, ext: float = 0.25, bg_size: int = 31,
           seed_contrast: float | None = None) -> list[dict]:
    flat = img - np.median(img, axis=0, keepdims=True)
    resid = gaussian_filter(np.abs(flat - median_filter(flat, size=bg_size)), 1)
    mu, sd = resid.mean(), resid.std()
    lab, n = label(binary_dilation(resid > mu + k_lo * sd, iterations=dil))
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_area:
            continue
        peak = float(resid[comp].max())
        if seed_contrast is not None:
            if peak < seed_contrast:
                continue
            score = peak
        else:
            if peak <= mu + k_hi * sd:
                continue
            score = peak / (mu + k_hi * sd)
        core = comp & (resid > ext * peak)
        ys, xs = np.where(core if core.any() else comp)
        out.append({"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "score": float(score)})
    return out


def detect_fod(img: np.ndarray, **kw) -> list[dict]:
    params = {"seed_contrast": 0.2, "k_lo": 1.5, "min_area": 12, "dil": 2,
              "ext": 0.3, "bg_size": 21}
    params.update(kw)
    return detect(img, **params)


def iou(a, b) -> float:
    ax = b["bbox"] if isinstance(b, dict) else b
    a = a["bbox"] if isinstance(a, dict) else a
    ix0, iy0 = max(a[0], ax[0]), max(a[1], ax[1])
    ix1, iy1 = min(a[2], ax[2]), min(a[3], ax[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = lambda z: max((z[2] - z[0]) * (z[3] - z[1]), 1e-9)  # noqa: E731
    return inter / (area(a) + area(ax) - inter)


def mean_average_precision(all_preds: list[list[dict]], all_gts: list[list],
                           thr: float = 0.5) -> float:
    """101-point interpolated AP with greedy IoU matching — works identically on
    YOLO outputs, which is the point."""
    recs = sorted(((p["score"], i, p["bbox"]) for i, ps in enumerate(all_preds) for p in ps),
                  key=lambda t: -t[0])
    n_gt = sum(len(g) for g in all_gts)
    matched = [set() for _ in all_gts]
    tp, fp = np.zeros(len(recs)), np.zeros(len(recs))
    for r, (_s, i, bb) in enumerate(recs):
        cand = [(iou(bb, g), j) for j, g in enumerate(all_gts[i]) if j not in matched[i]]
        best, bj = max(cand, default=(0.0, -1))
        if best >= thr:
            tp[r] = 1
            matched[i].add(bj)
        else:
            fp[r] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0
                          for t in np.linspace(0, 1, 101)]))
