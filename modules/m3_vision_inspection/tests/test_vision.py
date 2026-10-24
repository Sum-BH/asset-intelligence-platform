"""Ported from NB13-NB18 in-notebook asserts."""
import numpy as np
from aip_vision.anomaly import PatchCoreLite, evaluate, patch_features
from aip_vision.defects import detect, detect_fod, iou, mean_average_precision
from aip_vision.schema import VisionFinding, describe_rule_based, to_evidence_fields
from aip_vision.synthetic import (
    add_fod,
    base_texture,
    inject_defect,
    make_anomaly_set,
    make_detection_set,
    tarmac,
    thermal_frame,
)
from aip_vision.thermal import detect_hotspots, severity_band


def test_gt_boxes_derived_from_changed_pixels():
    img = base_texture(1)
    out, bbox = inject_defect(img, "blob", np.random.default_rng(2))
    ys, xs = np.where(np.abs(out - img) > 0.05)
    assert bbox == [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def test_patch_features_finite():
    f = patch_features(base_texture(3))
    assert f.ndim == 2 and f.shape[1] == 8 and np.all(np.isfinite(f))


def test_anomaly_auroc_gate():
    train, test, y = make_anomaly_set(category_seed=0)
    res = evaluate(PatchCoreLite().fit(train), test, y)
    assert res["auroc"] >= 0.95


def test_detection_map_gate():
    imgs, gts = make_detection_set(n=20)
    preds = [detect(im) for im in imgs]
    assert mean_average_precision(preds, gts, 0.5) >= 0.55


def test_thermal_recall_gate():
    hits = tot = 0
    for i in range(15):
        fr, gts = thermal_frame(9000 + i)
        dets = detect_hotspots(fr)
        for g in gts:
            tot += 1
            hits += any(iou(g["bbox"], d["bbox"]) >= 0.5 for d in dets)
    assert hits / tot >= 0.85


def test_severity_bands():
    assert severity_band(50) == "immediate"
    assert severity_band(20) == "repair_soon"
    assert severity_band(8) == "plan_repair"
    assert severity_band(2) == "monitor"


def test_fod_detection():
    imgs, gts = [], []
    for i in range(20):
        r = np.random.default_rng(6000 + i)
        img, _ = tarmac(6000 + i)
        boxes = []
        for _ in range(int(r.integers(1, 3))):
            img, g = add_fod(img, r)
            boxes.append(g)
        imgs.append(img)
        gts.append(boxes)
    preds = [detect_fod(im) for im in imgs]
    assert mean_average_precision(preds, gts, 0.5) >= 0.60


def test_vision_finding_describe_only():
    d = describe_rule_based("surface defect", [1, 2, 10, 12], 1.5, "rgb")
    assert "no cause inferred" in d.lower()


def test_evidence_mapping_roundtrip():
    fs = [VisionFinding(finding_id="v1", asset_id="a", modality="rgb", image_ref="x",
                        defect_type="surface_defect", score=1.2, bbox=[0, 0, 5, 5],
                        description=describe_rule_based("d", [0, 0, 5, 5], 1.2, "rgb"),
                        model_name="m", model_version="v")]
    ev = to_evidence_fields(fs)
    assert ev["top_features"][0].split(":")[1] == "surface_defect"
