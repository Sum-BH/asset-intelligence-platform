"""PatchCore-lite visual anomaly detection (promoted from NB13/NB14)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

PATCH, STRIDE = 16, 8


def patch_features(img: np.ndarray, patch: int = PATCH, stride: int = STRIDE) -> np.ndarray:
    gm = np.hypot(sobel(img, 0), sobel(img, 1))
    size = img.shape[0]
    feats = []
    for y in range(0, size - patch + 1, stride):
        for x in range(0, size - patch + 1, stride):
            p, g = img[y:y + patch, x:x + patch], gm[y:y + patch, x:x + patch]
            feats.append([p.mean(), p.std(), p.min(), p.max(), g.mean(), g.std(),
                          np.percentile(p, 25), np.percentile(p, 75)])
    return np.array(feats)


class PatchCoreLite:
    """Memory-bank anomaly detection: kNN distance of patch features to a bank of
    healthy patches. Same principle as PatchCore, classical features in place of a
    CNN backbone — swap in anomalib on the workstation."""

    def fit(self, train_imgs):
        bank = np.vstack([patch_features(im) for im in train_imgs])
        self.nn = NearestNeighbors(n_neighbors=1).fit(bank)
        return self

    def patch_distances(self, img: np.ndarray) -> np.ndarray:
        d, _ = self.nn.kneighbors(patch_features(img))
        return d

    def image_score(self, img: np.ndarray, percentile: float = 99) -> float:
        return float(np.percentile(self.patch_distances(img), percentile))


def evaluate(model: PatchCoreLite, test_imgs, labels) -> dict:
    scores = np.array([model.image_score(im) for im in test_imgs])
    auroc = float(roc_auc_score(labels, scores))
    cands = np.quantile(scores, np.linspace(0.05, 0.95, 37))
    f1s = [f1_score(labels, scores >= t) for t in cands]
    best = int(np.argmax(f1s))
    return {"auroc": round(auroc, 4), "threshold": float(cands[best]),
            "f1_at_threshold": round(float(f1s[best]), 3)}
