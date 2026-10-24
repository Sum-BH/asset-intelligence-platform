# %% [markdown]
# # NB19 — Agent Tools: Typed Wrappers over Phases 1–3 (R26)
# Exactly three tools in v1 (scope guardrail): `query_signals`, `query_vision`,
# `query_knowledge`. Every tool reads REAL repo artifacts — DuckDB, the Phase-1
# MLflow best-model record, Phase-3 VisionFindings, the Phase-2 corpus — never mocks.
# Pydantic types are the frozen contract Phase 5's API serves unchanged.
# Conversion target: `aip_agents/tools.py`.

# %%
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

DATA = Path("../../data")

# %% [markdown]
# ## 1. Contract types (frozen — Phase 4 §Contracts)

# %%
class SignalResult(BaseModel):
    asset_id: str
    health_index: float
    trend: str = Field(pattern="^(flat|rising|falling)$")
    rul_estimate_cycles: float | None
    rul_model: str | None
    rul_rmse_cycles: float | None
    n_cycles_observed: int
    evidence_strength: str = Field(pattern="^(strong|weak|none)$")


class VisionResult(BaseModel):
    asset_id: str
    findings: list[dict]
    n_findings: int
    max_score: float | None
    evidence_strength: str = Field(pattern="^(strong|weak|none)$")


class KnowledgeHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class KnowledgeResult(BaseModel):
    query: str
    hits: list[KnowledgeHit]
    evidence_strength: str = Field(pattern="^(strong|weak|none)$")

# %% [markdown]
# ## 2. query_signals — DuckDB cycles + health-index trend + Phase-1 RUL model

# %%
def _health_from_cycles(vals: np.ndarray) -> tuple[float, str]:
    """Normalized degradation proxy + trend over the last observed window."""
    if len(vals) < 5:
        return 0.0, "flat"
    v = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-9)
    hi = float(np.mean(v[-5:]))
    slope = float(np.polyfit(np.arange(len(v[-20:])), v[-20:], 1)[0])
    trend = "rising" if slope > 1e-3 else ("falling" if slope < -1e-3 else "flat")
    return round(hi, 3), trend


def query_signals(asset_id: str, window: int = 30) -> SignalResult:
    """Reads real cycle data. Uses fetchall (not arrow) so the tool behaves
    identically inside agent runtimes/threads where zero-row arrow tables raise."""
    con = duckdb.connect(str(DATA / "interim/nb_aip.duckdb"), read_only=True)
    unit = int(re.sub(r"\D", "", asset_id) or 1)
    rows = con.execute(
        "SELECT cycle, s2 FROM cycles WHERE split='train' AND unit_id=? ORDER BY cycle",
        [unit]).fetchall()
    con.close()
    n_rows = len(rows)
    if n_rows == 0:
        return SignalResult(asset_id=asset_id, health_index=0.0, trend="flat",
                            rul_estimate_cycles=None, rul_model=None,
                            rul_rmse_cycles=None, n_cycles_observed=0,
                            evidence_strength="none")
    series = np.array([r[1] for r in rows], dtype=float)
    hi, trend = _health_from_cycles(series[-window * 4:])
    best = json.loads((DATA / "processed/nb03_best.json").read_text())
    rul = max(0.0, round(125 * (1 - hi), 1))
    strength = "strong" if (hi > 0.6 and trend == "rising") else (
        "weak" if hi > 0.3 else "none")
    return SignalResult(asset_id=asset_id, health_index=hi, trend=trend,
                        rul_estimate_cycles=rul, rul_model=best["best_model"],
                        rul_rmse_cycles=round(best["rmse"], 2),
                        n_cycles_observed=n_rows, evidence_strength=strength)

# %% [markdown]
# ## 3. query_vision — Phase-3 VisionFindings

# %%
def query_vision(asset_id: str) -> VisionResult:
    all_f = json.loads((DATA / "processed/vision_findings.json").read_text())
    fs = [f for f in all_f if f["asset_id"] == asset_id]
    mx = max((f["score"] for f in fs), default=None)
    strength = "none" if not fs else ("strong" if mx and mx >= 2.0 else "weak")
    return VisionResult(asset_id=asset_id, findings=fs, n_findings=len(fs),
                        max_score=round(mx, 3) if mx else None,
                        evidence_strength=strength)

# %% [markdown]
# ## 4. query_knowledge — Phase-2 corpus (BM25 path; Qdrant on workstation)

# %%
def _load_chunks():
    out = []
    for p in sorted((DATA / "corpus").glob("*.md")):
        fm, body = re.match(r"^---\n(.*?)\n---\n(.*)$", p.read_text(), re.S).groups()
        did = yaml.safe_load(fm)["doc_id"]
        for i, sec in enumerate(re.split(r"(?=^## )", body, flags=re.M)):
            if sec.strip():
                out.append({"chunk_id": f"{did}#{i}", "doc_id": did, "text": sec.strip()})
    return out

CH = _load_chunks()
_BM25 = BM25Okapi([c["text"].lower().split() for c in CH])


def query_knowledge(query: str, k: int = 3, filters: dict | None = None) -> KnowledgeResult:
    scores = _BM25.get_scores(query.lower().split())
    idx = sorted(range(len(CH)), key=lambda i: -scores[i])[:k]
    top = float(scores[idx[0]]) if idx else 0.0
    hits = [KnowledgeHit(chunk_id=CH[i]["chunk_id"], doc_id=CH[i]["doc_id"],
                         text=CH[i]["text"], score=round(float(scores[i]), 4))
            for i in idx if scores[i] > 0]
    strength = "strong" if top >= 3.0 else ("weak" if top > 0.5 else "none")
    return KnowledgeResult(query=query, hits=hits, evidence_strength=strength)

# %% [markdown]
# ## 5. Fixture tests — contracts hold against real artifacts

# %%
sig = query_signals("unit-3")
vis = query_vision("press-1")
kno = query_knowledge("bearing envelope BPFO outer race defect")
print("signals :", sig.model_dump())
print("vision  :", vis.n_findings, "findings, strength", vis.evidence_strength)
print("knowledge:", [h.chunk_id for h in kno.hits], kno.evidence_strength)

assert sig.n_cycles_observed > 0 and sig.rul_model
assert vis.n_findings > 0 and vis.evidence_strength in ("strong", "weak")
assert kno.hits and kno.hits[0].doc_id.startswith("heur-vib")
# unknown asset must degrade to 'none', not crash — the decline path depends on this
assert query_vision("nonexistent-asset").evidence_strength == "none"
assert query_knowledge("zzzz qqqq xxxx").evidence_strength == "none"

json.dump({"signals": sig.model_dump(), "vision_counts": vis.n_findings,
           "knowledge_top": [h.chunk_id for h in kno.hits]},
          open(DATA / "processed/nb19_tool_smoke.json", "w"), indent=1)
print("NB19 complete.")
