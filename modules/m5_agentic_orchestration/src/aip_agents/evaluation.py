"""Agent evaluation metrics (promoted from NB23)."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[([\w#-]+)\]")


def action_grounded(work_order: dict, valid_chunk_ids: set[str]) -> bool:
    """Every raised work order must cite real knowledge-base chunks."""
    if work_order["status"] == "declined":
        return True
    cited = _CITATION.findall(work_order["recommended_action"])
    return bool(cited) and all(c in valid_chunk_ids for c in cited)


def evidence_consistent(work_order: dict) -> bool:
    """Priority must not exceed what the evidence supports."""
    strength = work_order["evidence"].get("evidence_strength", "none")
    if work_order["status"] == "declined":
        return work_order["priority"] == 1 and work_order["duration_h_est"] == 0
    return not (work_order["priority"] == 3 and strength != "strong")


def summarize(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {"n_scenarios": len(rows),
            "expectation_match": round(sum(r["expect_ok"] for r in rows) / n, 3),
            "groundedness": round(sum(r["grounded"] for r in rows) / n, 3),
            "evidence_consistency": round(sum(r["consistent"] for r in rows) / n, 3),
            "declines": sum(r["status"] == "declined" for r in rows)}
