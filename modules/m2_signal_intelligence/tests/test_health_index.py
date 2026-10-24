import numpy as np
from aip_signal.health_index import health_index, kalman_smooth, window_features


def _synthetic_run_to_failure(n_windows=120, win=2048, fs=20000.0, seed=11):
    """Healthy sine+noise; from 60% onward inject growing impulsive fault."""
    rng = np.random.default_rng(seed)
    t = np.arange(win) / fs
    windows = []
    for i in range(n_windows):
        x = 0.5 * np.sin(2 * np.pi * 157 * t) + rng.normal(0, 0.1, win)
        if i > int(n_windows * 0.6):
            sev = (i - n_windows * 0.6) / (n_windows * 0.4)
            impulses = rng.random(win) < 0.002 * (1 + 5 * sev)
            x = x + impulses * rng.normal(0, 3.0 * sev, win)
        windows.append(x)
    return np.array(windows), fs


def test_window_features_shape():
    x = np.random.default_rng(0).normal(size=2048)
    f = window_features(x, fs=20000.0)
    assert f.shape == (12,) and np.all(np.isfinite(f))


def test_kalman_smooth_reduces_variance():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 500)
    s = kalman_smooth(z)
    assert s.std() < z.std() * 0.5


def test_health_index_rises_before_failure():
    windows, fs = _synthetic_run_to_failure()
    hi = health_index(windows, fs)
    healthy = hi[: int(len(hi) * 0.5)]
    failing = hi[int(len(hi) * 0.85) :]
    assert failing.mean() > healthy.mean() + 3 * healthy.std(), (
        "health index must clearly separate failing tail from healthy head"
    )
