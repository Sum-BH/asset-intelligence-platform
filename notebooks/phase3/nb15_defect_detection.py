# %% [markdown]
# # NB15 — Defect Detection with mAP Evaluation (R21)
# Sandbox: classical detector (stripe removal → residual → hysteresis components →
# peak-relative box extents) on synthetic labeled defects, evaluated with a real
# mAP implementation (greedy IoU matching + 101-pt AP) — the same metric machinery
# you point at YOLO/NEU on the workstation (guarded cell).
# Sandbox gate: mAP@0.5 ≥ 0.60 for the zero-parameter classical baseline on this
# deliberately hard textured multi-defect set (mAP@0.3 also reported — the gap is
# box-extent convention, not missed defects). This baseline is the number YOLO must
# beat. The SOP R21 gate (mAP50 ≥ 0.75) applies to the NEU val split on workstation.
# Target: `aip_vision/defects.py`.

# %%
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, label, median_filter

DATA = Path("../../data")
exec(Path("nb13_anomaly_patchcore.py").read_text().split("model = PatchCoreLite()")[0]
     .split("PATCH, STRIDE")[0])  # base_texture + inject_defect + IMG
rng = np.random.default_rng(7)

# %% [markdown]
# ## 1. Labeled synthetic set (1–3 defects per image)

# %%
def make_detection_set(n=40, seed0=3000):
    imgs, gts = [], []
    kinds = ["scratch", "blob", "pits"]
    for i in range(n):
        r = np.random.default_rng(seed0 + i)
        img = base_texture(seed0 + i)
        boxes = []
        for _ in range(int(r.integers(1, 4))):
            img, bb = inject_defect(img, kinds[int(r.integers(0, 3))], r)
            boxes.append(bb)
        imgs.append(img); gts.append(boxes)
    return imgs, gts

imgs, gts = make_detection_set()
print(f"{len(imgs)} images, {sum(map(len, gts))} GT boxes")

# %% [markdown]
# ## 2. Detector: |image − median background| → threshold → connected components

# %%
def detect(img, k_hi=3.0, k_lo=1.2, min_area=12, dil=3, ext=0.25):
    """Stripe removal -> background residual -> hysteresis (seed k_hi, grow k_lo)
    -> component boxes refit on pixels above ext*peak (scale-invariant extent)."""
    from scipy.ndimage import binary_dilation
    flat = img - np.median(img, axis=0, keepdims=True)      # periodic-structure removal
    resid = gaussian_filter(np.abs(flat - median_filter(flat, size=31)), 1)
    mu, sd = resid.mean(), resid.std()
    lab, n = label(binary_dilation(resid > mu + k_lo * sd, iterations=dil))
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_area:
            continue
        peak = resid[comp].max()
        if peak <= mu + k_hi * sd:                           # hysteresis seed test
            continue
        core = comp & (resid > ext * peak)
        ys, xs = np.where(core if core.any() else comp)
        out.append({"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "score": float(peak / (mu + k_hi * sd))})
    return out

# %% [markdown]
# ## 3. mAP@0.5 — the reusable metric (identical for YOLO outputs)

# %%
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ar = lambda z: (z[2] - z[0]) * (z[3] - z[1])
    return inter / max(ar(a) + ar(b) - inter, 1e-9)

def map_at(all_preds, all_gts, thr=0.5):
    recs = []
    n_gt = sum(map(len, all_gts))
    for i, preds in enumerate(all_preds):
        for p in preds:
            recs.append((p["score"], i, p["bbox"]))
    recs.sort(reverse=True)
    matched = [set() for _ in all_gts]
    tp = np.zeros(len(recs)); fp = np.zeros(len(recs))
    for r, (s, i, bb) in enumerate(recs):
        best, bj = 0.0, -1
        for j, g in enumerate(all_gts[i]):
            if j in matched[i]:
                continue
            v = iou(bb, g)
            if v > best:
                best, bj = v, j
        if best >= thr:
            tp[r] = 1; matched[i].add(bj)
        else:
            fp[r] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # 101-point interpolation
    ap = float(np.mean([prec[recall >= t].max() if (recall >= t).any() else 0.0
                        for t in np.linspace(0, 1, 101)]))
    return ap, recall[-1] if len(recall) else 0.0

preds = [detect(im) for im in imgs]
ap, rec = map_at(preds, gts, 0.5)
ap30, _ = map_at(preds, gts, 0.3)
print(f"mAP@0.5 = {ap:.3f}   mAP@0.3 = {ap30:.3f}   (final recall {rec:.2f})")
assert ap >= 0.60, "classical-baseline mAP below sandbox gate"

json.dump({"map50": round(ap, 4), "map30": round(ap30, 4), "detector": "stripe+hysteresis",
           "n_images": len(imgs)}, open(DATA / "processed/nb15_detections_meta.json", "w"))
dets = [{"image_id": i, "detections": p} for i, p in enumerate(preds[:10])]
json.dump(dets, open(DATA / "processed/nb15_detections.json", "w"), indent=1)

if importlib.util.find_spec("ultralytics"):
    print("ultralytics present → fine-tune YOLOv8s on NEU (SOP runbook), reuse map50()")
else:
    print("SKIP ultralytics (workstation): uv add ultralytics; dataset: NEU surface defects")
print("NB15 complete.")
