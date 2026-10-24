"""VisionFinding — frozen contract consumed by aip_agents.query_vision (from NB18)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


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
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def describe_rule_based(kind: str, bbox: list[int], score: float, modality: str) -> str:
    """Describe-only discipline: visible extent, no cause speculation."""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    unit = "degC peak-above-background" if modality == "thermal" else "relative confidence"
    return (f"{kind} region approximately {w}x{h}px at ({bbox[0]},{bbox[1]}); "
            f"{score:.2f} {unit}. Visible extent only; no cause inferred.")


def to_evidence_fields(findings: list[VisionFinding]) -> dict:
    """Frozen mapping into the Phase-1 Evidence schema."""
    return {"top_features": [f"vision:{f.defect_type}:{f.score:.2f}" for f in findings[:3]],
            "notes": " | ".join(f.description for f in findings[:2])}
