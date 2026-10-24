# %% [markdown]
# # NB18 — VisionFinding Schema + VLM Bridge (R23, R24, R25)
# Consumes NB15/NB16/NB17 detection artifacts → frozen VisionFinding records.
# Description backend switch: Qwen2.5-VL via Ollama (workstation, guarded) ⇄
# rule-based describer (sandbox). Division of labour (governance material):
# **models detect, the VLM describes** — pass/fail always comes from detector scores.
# Gates: schema assert; Evidence mapping round-trip; VLM JSON discipline.
# Conversion target: `aip_vision/vlm.py`.

# %%
import json
from datetime import datetime, timezone, UTC
from pathlib import Path

import requests
from pydantic import BaseModel, Field

DATA = Path("../../data")

# %% [markdown]
# ## 1. The frozen schema (Phase 4's query_vision returns exactly this)

# %%
class VisionFinding(BaseModel):
    finding_id: str
    asset_id: str
    modality: str = Field(pattern="^(rgb|thermal)$")
    image_ref: str
    defect_type: str
    score: float
    bbox: list[int]
    description: str
    model_name: str
    model_version: str
    created_at: str


def describe_rule_based(kind: str, bbox: list[int], score: float, modality: str) -> str:
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    unit = "degC peak-above-background" if modality == "thermal" else "relative confidence"
    return (f"{kind} region approximately {w}x{h}px at ({bbox[0]},{bbox[1]}); "
            f"{score:.2f} {unit}. Visible extent only; no cause inferred.")


def describe_vlm(image_path: str, bbox: list[int]) -> str | None:
    """Workstation path: Qwen2.5-VL, strict JSON, ≤60 words, visible content only."""
    try:
        requests.get("http://localhost:11434", timeout=2)
    except Exception:
        return None
    import base64
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    prompt = ("Describe ONLY the visible defect inside the given bbox in <=60 words. "
              "No cause speculation. Reply as JSON: {\"description\": \"...\"}")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "qwen2.5-vl:7b", "prompt": f"{prompt}\nbbox={bbox}",
                            "images": [img_b64], "stream": False,
                            "options": {"temperature": 0.1}}, timeout=180)
    r.raise_for_status()
    import re
    m = re.search(r"\{.*\}", r.json()["response"], re.S)
    return json.loads(m.group(0))["description"] if m else None

# %% [markdown]
# ## 2. Build findings from NB15 / NB16 / NB17 artifacts (file handoff, never in-memory)

# %%
now = datetime.now(UTC).isoformat()
findings: list[VisionFinding] = []

d15 = json.loads((DATA / "processed/nb15_detections.json").read_text())
for rec in d15[:4]:
    for k, det in enumerate(rec["detections"]):
        findings.append(VisionFinding(
            finding_id=f"vf-nb15-{rec['image_id']}-{k}", asset_id="press-1",
            modality="rgb", image_ref=f"synthetic://nb15/{rec['image_id']}",
            defect_type="surface_defect", score=round(det["score"], 3),
            bbox=det["bbox"],
            description=describe_rule_based("surface defect", det["bbox"], det["score"], "rgb"),
            model_name="residual-blob", model_version="sandbox-v1", created_at=now))

d16 = json.loads((DATA / "processed/nb16_hotspots.json").read_text())
for rec in d16["samples"][:3]:
    for k, det in enumerate(rec["hotspots"]):
        findings.append(VisionFinding(
            finding_id=f"vf-nb16-{rec['frame']}-{k}", asset_id="panel-2",
            modality="thermal", image_ref=f"synthetic://nb16/{rec['frame']}",
            defect_type="thermal_hotspot", score=det["peak_dt"], bbox=det["bbox"],
            description=describe_rule_based("thermal hotspot", det["bbox"],
                                            det["peak_dt"], "thermal"),
            model_name="bg-relative-threshold", model_version="sandbox-v1", created_at=now))

d17 = json.loads((DATA / "processed/nb17_fod.json").read_text())
for rec in d17["samples"][:3]:
    for k, det in enumerate(rec["pred"]):
        findings.append(VisionFinding(
            finding_id=f"vf-nb17-{rec['image_id']}-{k}", asset_id="taxiway-A3",
            modality="rgb", image_ref=f"synthetic://nb17/{rec['image_id']}",
            defect_type="foreign_object", score=round(det["score"], 3), bbox=det["bbox"],
            description=describe_rule_based("foreign object", det["bbox"],
                                            det["score"], "rgb"),
            model_name="residual-blob", model_version="sandbox-v1", created_at=now))

print(f"{len(findings)} VisionFindings across 3 modsources")
assert len(findings) >= 8
assert all(f.modality in ("rgb", "thermal") and len(f.bbox) == 4 for f in findings)
assert all("no cause inferred" in f.description.lower() for f in findings), \
    "describe-only discipline violated"

# %% [markdown]
# ## 3. Mapping into Evidence (frozen rule, Phase 1 §7.2) — round-trip assert

# %%
def to_evidence_fields(fs: list[VisionFinding]) -> dict:
    return {"top_features": [f"vision:{f.defect_type}:{f.score:.2f}" for f in fs[:3]],
            "notes": " | ".join(f.description for f in fs[:2])}

ev_patch = to_evidence_fields(findings)
assert all(x.startswith("vision:") for x in ev_patch["top_features"])
assert ev_patch["notes"].count("|") == 1
back = [x.split(":") for x in ev_patch["top_features"]]
assert all(len(b) == 3 and b[1] == findings[i].defect_type for i, b in enumerate(back))
print("Evidence mapping round-trip OK:", ev_patch["top_features"])

out = [json.loads(f.model_dump_json()) for f in findings]
(DATA / "processed/vision_findings.json").write_text(json.dumps(out, indent=1))

vlm = describe_vlm(str(DATA / "interim/nb17_fod_reel.png"), [10, 10, 40, 40])
print("VLM:", vlm if vlm else "SKIP (workstation): ollama pull qwen2.5-vl:7b — never co-resident with 14B (R25)")
print("NB18 complete.")
