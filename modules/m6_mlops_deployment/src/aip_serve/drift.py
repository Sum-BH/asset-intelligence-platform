"""Drift monitoring: feature PSI + prediction KS (promoted from NB27)."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

PSI_ALERT = 0.2
KS_ALERT = 0.01


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def check_batch(reference: dict[str, np.ndarray], batch: dict[str, np.ndarray],
                preds_ref: np.ndarray | None = None, preds_cur: np.ndarray | None = None,
                psi_alert: float = PSI_ALERT, ks_alert: float = KS_ALERT) -> list[dict]:
    alerts = []
    for feature, ref_vals in reference.items():
        if feature not in batch:
            continue
        value = psi(ref_vals, batch[feature])
        if value > psi_alert:
            alerts.append({"kind": "feature_psi", "feature": feature,
                           "value": round(value, 4)})
    if preds_ref is not None and preds_cur is not None:
        p = float(ks_2samp(preds_ref, preds_cur).pvalue)
        if p < ks_alert:
            alerts.append({"kind": "prediction_ks", "feature": "rul_pred",
                           "value": round(p, 6)})
    return alerts
