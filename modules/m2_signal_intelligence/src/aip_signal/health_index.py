"""Vibration health index for run-to-failure data (IMS-style).

Per time-window features (FFT band energies + envelope-spectrum energy +
time-domain stats) -> Isolation Forest trained on the healthy head of the
recording -> anomaly score over time -> 1-D Kalman smoothing.

The health index should stay flat while healthy and rise ahead of failure.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from sklearn.ensemble import IsolationForest

TIME_FEATS = ["rms", "kurtosis", "crest", "env_rms"]


def window_features(x: np.ndarray, fs: float, n_bands: int = 8) -> np.ndarray:
    """Feature vector for one vibration window: time stats + FFT band energies."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2))
    kurt = float(np.mean((x / (x.std() + 1e-12)) ** 4))
    crest = float(np.max(np.abs(x)) / (rms + 1e-12))
    env = np.abs(hilbert(x))
    env_rms = float(np.sqrt(np.mean((env - env.mean()) ** 2)))

    spec = np.abs(np.fft.rfft(x)) ** 2
    bands = np.array_split(spec[1:], n_bands)  # skip DC
    band_energy = np.log10(np.array([b.sum() for b in bands]) + 1e-12)
    return np.concatenate([[rms, kurt, crest, env_rms], band_energy])


def kalman_smooth(z: np.ndarray, q: float = 1e-4, r: float = 1e-2) -> np.ndarray:
    """1-D constant-level Kalman filter (process var q, measurement var r)."""
    x_hat = z[0]
    p = 1.0
    out = np.empty_like(z, dtype=float)
    for i, meas in enumerate(z):
        p = p + q  # predict
        k = p / (p + r)  # gain
        x_hat = x_hat + k * (meas - x_hat)
        p = (1 - k) * p
        out[i] = x_hat
    return out


def health_index(
    windows: np.ndarray, fs: float, healthy_fraction: float = 0.3, random_state: int = 42
) -> np.ndarray:
    """Smoothed anomaly score per window; higher = less healthy.

    windows: 2-D array (n_windows, window_len), chronological order.
    The Isolation Forest is fit ONLY on the first `healthy_fraction` of
    windows — the run-to-failure assumption that the recording starts healthy.
    """
    feats = np.vstack([window_features(w, fs) for w in windows])
    n_healthy = max(int(len(feats) * healthy_fraction), 10)
    forest = IsolationForest(n_estimators=200, random_state=random_state)
    forest.fit(feats[:n_healthy])
    raw = -forest.score_samples(feats)  # higher = more anomalous
    baseline = np.median(raw[:n_healthy])
    return kalman_smooth(raw - baseline)
