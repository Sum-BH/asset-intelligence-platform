"""The three agent tools (promoted from NB19). Scope guardrail: exactly three in v1.

Tools read REAL artifacts. `query_signals` uses fetchall rather than arrow because
zero-row arrow tables raise inside agent runtimes (found in NB20).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
from aip_core import paths
from aip_rag.retrieve import BM25Retriever, evidence_strength
from pydantic import BaseModel, Field


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


def health_from_series(vals: np.ndarray) -> tuple[float, str]:
    if len(vals) < 5:
        return 0.0, "flat"
    v = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-9)
    hi = float(np.mean(v[-5:]))
    slope = float(np.polyfit(np.arange(len(v[-20:])), v[-20:], 1)[0])
    trend = "rising" if slope > 1e-3 else ("falling" if slope < -1e-3 else "flat")
    return round(hi, 3), trend


def query_signals(asset_id: str, window: int = 30, db_path: str | Path | None = None,
                  best_model_path: str | Path | None = None) -> SignalResult:
    db_path = Path(db_path or paths.interim / "nb_aip.duckdb")
    unit = int(re.sub(r"\D", "", asset_id) or 1)
    rows = []
    if db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute("SELECT cycle, s2 FROM cycles WHERE split='train' AND unit_id=? "
                           "ORDER BY cycle", [unit]).fetchall()
        con.close()
    if not rows:
        return SignalResult(asset_id=asset_id, health_index=0.0, trend="flat",
                            rul_estimate_cycles=None, rul_model=None, rul_rmse_cycles=None,
                            n_cycles_observed=0, evidence_strength="none")
    hi, trend = health_from_series(np.array([r[1] for r in rows], dtype=float)[-window * 4:])
    bm = Path(best_model_path or paths.processed / "nb03_best.json")
    best = json.loads(bm.read_text()) if bm.exists() else {"best_model": "unknown", "rmse": 0.0}
    strength = "strong" if (hi > 0.6 and trend == "rising") else ("weak" if hi > 0.3 else "none")
    return SignalResult(asset_id=asset_id, health_index=hi, trend=trend,
                        rul_estimate_cycles=max(0.0, round(125 * (1 - hi), 1)),
                        rul_model=best["best_model"], rul_rmse_cycles=round(best["rmse"], 2),
                        n_cycles_observed=len(rows), evidence_strength=strength)


def query_vision(asset_id: str, findings_path: str | Path | None = None) -> VisionResult:
    p = Path(findings_path or paths.processed / "vision_findings.json")
    all_f = json.loads(p.read_text()) if p.exists() else []
    fs = [f for f in all_f if f["asset_id"] == asset_id]
    mx = max((f["score"] for f in fs), default=None)
    strength = "none" if not fs else ("strong" if mx and mx >= 2.0 else "weak")
    return VisionResult(asset_id=asset_id, findings=fs, n_findings=len(fs),
                        max_score=round(mx, 3) if mx else None, evidence_strength=strength)


_RETRIEVER: BM25Retriever | None = None


def get_retriever(corpus_dir: str | Path | None = None) -> BM25Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = BM25Retriever(corpus_dir or paths.data / "corpus")
    return _RETRIEVER


def query_knowledge(query: str, k: int = 3, corpus_dir: str | Path | None = None
                    ) -> KnowledgeResult:
    hits = get_retriever(corpus_dir).search(query, k=k)
    top = hits[0].score if hits else 0.0
    return KnowledgeResult(query=query,
                           hits=[KnowledgeHit(**h.as_dict()) for h in hits],
                           evidence_strength=evidence_strength(top))
