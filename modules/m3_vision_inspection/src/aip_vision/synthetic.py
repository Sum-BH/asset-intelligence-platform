"""Synthetic imagery generators (promoted from NB13/NB16/NB17).

Lesson encoded here: ground-truth boxes are derived from ACTUALLY CHANGED PIXELS,
never from hand-drawn margins. Mismatched GT conventions cap IoU permanently and
look like detector failure.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

IMG = 128
FLAT_SENSOR_LIKE = {0, 4, 5, 9, 15, 17, 18}
DEFECT_KINDS = ("scratch", "blob", "pits")
FOD_KINDS = ("bolt", "wrench", "fragment")


def base_texture(seed: int, size: int = IMG) -> np.ndarray:
    r = np.random.default_rng(seed)
    tex = gaussian_filter(r.normal(0, 1, (size, size)), 3)
    _, xx = np.mgrid[0:size, 0:size]
    tex += 0.3 * np.sin(xx / 6.0 + r.uniform(0, 6))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return 0.35 + 0.3 * tex


def _changed_bbox(after: np.ndarray, before: np.ndarray, tol: float) -> list[int]:
    ys, xs = np.where(np.abs(after - before) > tol)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def inject_defect(img: np.ndarray, kind: str, r: np.random.Generator
                  ) -> tuple[np.ndarray, list[int]]:
    size = img.shape[0]
    im = img.copy()
    if kind == "scratch":
        x0, y0 = r.integers(15, size - 60, 2)
        length, ang = r.integers(35, 60), r.uniform(0, np.pi)
        xs = (x0 + np.cos(ang) * np.arange(length)).astype(int)
        ys = (y0 + np.sin(ang) * np.arange(length)).astype(int)
        for dx in (-1, 0, 1):
            im[np.clip(ys + dx, 0, size - 1), np.clip(xs, 0, size - 1)] -= 0.28
    elif kind == "blob":
        cx, cy = r.integers(25, size - 25, 2)
        rad = r.integers(6, 12)
        yy, xx = np.mgrid[0:size, 0:size]
        im += 0.35 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad**2)))
    else:
        cx, cy = r.integers(25, size - 25, 2)
        for _ in range(r.integers(6, 12)):
            px, py = cx + r.integers(-10, 10), cy + r.integers(-10, 10)
            im[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2] -= 0.3
    im = np.clip(im, 0, 1)
    return im, _changed_bbox(im, img, 0.05)


def make_anomaly_set(category_seed: int = 0, n_good_train: int = 40,
                     n_good_test: int = 15, n_bad_test: int = 15):
    train = [base_texture(category_seed * 1000 + i) for i in range(n_good_train)]
    test, labels = [], []
    for i in range(n_good_test):
        test.append(base_texture(category_seed * 1000 + 500 + i))
        labels.append(0)
    for i in range(n_bad_test):
        img = base_texture(category_seed * 1000 + 800 + i)
        img, _ = inject_defect(img, DEFECT_KINDS[i % 3], np.random.default_rng(900 + i))
        test.append(img)
        labels.append(1)
    return train, test, np.array(labels)


def make_detection_set(n: int = 40, seed0: int = 3000):
    imgs, gts = [], []
    for i in range(n):
        r = np.random.default_rng(seed0 + i)
        img = base_texture(seed0 + i)
        boxes = []
        for _ in range(int(r.integers(1, 4))):
            img, bb = inject_defect(img, DEFECT_KINDS[int(r.integers(0, 3))], r)
            boxes.append(bb)
        imgs.append(img)
        gts.append(boxes)
    return imgs, gts


def thermal_frame(seed: int, size: int = IMG, n_hot: int | None = None):
    """Returns (frame in degC, hotspots with GT boxes at >1 degC added heat)."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 30 + 4 * (xx / size) + gaussian_filter(r.normal(0, 0.6, (size, size)), 4)
    hots = []
    for _ in range(int(r.integers(1, 4)) if n_hot is None else n_hot):
        cx, cy = r.integers(15, size - 15, 2)
        rad, d_t = int(r.integers(4, 10)), float(r.uniform(6, 25))
        heat = d_t * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad**2)))
        base += heat
        ys2, xs2 = np.where(heat > 1.0)
        hots.append({"bbox": [int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())],
                     "delta_t": round(d_t, 1)})
    return base, hots


def tarmac(seed: int, size: int = IMG):
    r = np.random.default_rng(seed)
    t = gaussian_filter(r.normal(0, 1, (size, size)), 1.2) * 0.05 + 0.22
    t += (r.random((size, size)) < 0.01) * 0.12
    x0 = int(r.integers(30, 90))
    t[:, x0:x0 + 4] = 0.85
    return np.clip(t, 0, 1), x0


def add_fod(img: np.ndarray, r: np.random.Generator):
    size = img.shape[0]
    im = img.copy()
    kind = FOD_KINDS[int(r.integers(0, 3))]
    cx, cy = int(r.integers(12, size - 12)), int(r.integers(12, size - 12))
    if kind == "bolt":
        yy, xx = np.mgrid[0:size, 0:size]
        im[((xx - cx) ** 2 + (yy - cy) ** 2) < 16] = 0.75
    elif kind == "wrench":
        im[cy - 2:cy + 2, max(cx - 9, 0):cx + 9] = 0.7
        im[cy - 4:cy + 4, max(cx - 11, 0):cx - 7] = 0.7
    else:
        h, w = int(r.integers(3, 7)), int(r.integers(4, 9))
        im[cy:cy + h, cx:cx + w] = 0.65
    return im, {"bbox": _changed_bbox(im, img, 0.02), "kind": kind}
