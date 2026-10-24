"""LangGraph work-order agent with a structural human-in-the-loop gate (from NB20).

No code path reaches status 'approved' without a human decision: the graph is
compiled with interrupt_before=['approval'].
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from aip_agents.tools import query_knowledge, query_signals, query_vision

SKILL_MAP = {"press-1": "mechanical", "pump-3": "hydraulics", "pump-7": "hydraulics",
             "panel-2": "electrical", "taxiway-A3": "airside", "unit-3": "mechanical"}
DEFAULT_SKILL = "mechanical"


class WorkOrder(BaseModel):
    """Business work order (distinct from scheduler.WorkOrderSpec — see ADR-003)."""

    wo_id: str
    asset_id: str
    created_at: str
    evidence: dict
    findings: list[dict] = []
    citations: list[str] = []
    recommended_action: str
    required_skill: str
    duration_h_est: int
    priority: int = Field(ge=1, le=3)
    status: Literal["draft", "approved", "rejected", "declined", "scheduled"] = "draft"
    approver: str | None = None
    reviewer_note: str | None = None


class AgentState(TypedDict, total=False):
    asset_id: str
    signals: dict
    vision: dict
    knowledge: dict
    work_order: dict
    decision: str
    approver: str
    reviewer_note: str


def plan_from_rules(sig: dict, vis: dict, kno: dict) -> dict:
    cite = kno["hits"][0]["chunk_id"] if kno["hits"] else "none"
    pri = 3 if (sig["evidence_strength"] == "strong" and vis["n_findings"] > 0) else (
        2 if sig["evidence_strength"] == "strong" else 1)
    return {"action": (f"Inspect {sig['asset_id']}: health index {sig['health_index']:.2f} "
                       f"({sig['trend']}), RUL ~{sig['rul_estimate_cycles']:.0f} cycles; "
                       f"schedule before RUL minus RMSE [{cite}]"),
            "priority": pri, "duration_h": 6 if pri == 3 else (4 if pri == 2 else 2)}


def _new_id() -> str:
    return "WO-" + uuid.uuid4().hex[:6]


def build_graph(planner=plan_from_rules, checkpointer=None):
    def n_signals(s: AgentState) -> AgentState:
        return {"signals": query_signals(s["asset_id"]).model_dump()}

    def n_vision(s: AgentState) -> AgentState:
        return {"vision": query_vision(s["asset_id"]).model_dump()}

    def n_knowledge(s: AgentState) -> AgentState:
        sig = s.get("signals", {})
        q = (f"{'rising' if sig.get('trend') == 'rising' else ''} health index degradation "
             f"action window RUL planning")
        return {"knowledge": query_knowledge(q).model_dump()}

    def n_fuse(s: AgentState) -> AgentState:
        sig, vis, kno = s["signals"], s["vision"], s["knowledge"]
        now = datetime.now(UTC).isoformat()
        skill = SKILL_MAP.get(s["asset_id"], DEFAULT_SKILL)
        if sig["evidence_strength"] == "none" and vis["evidence_strength"] == "none":
            wo = WorkOrder(wo_id=_new_id(), asset_id=s["asset_id"], created_at=now,
                           evidence=sig, recommended_action=(
                               "Insufficient evidence: no signal or vision support. "
                               "No work order raised."),
                           required_skill=skill, duration_h_est=0, priority=1,
                           status="declined")
            return {"work_order": wo.model_dump(), "decision": "declined"}
        plan = planner(sig, vis, kno)
        wo = WorkOrder(wo_id=_new_id(), asset_id=s["asset_id"], created_at=now,
                       evidence=sig, findings=vis["findings"][:3],
                       citations=[h["chunk_id"] for h in kno["hits"]],
                       recommended_action=plan["action"], priority=int(plan["priority"]),
                       duration_h_est=int(plan["duration_h"]), required_skill=skill,
                       status="draft")
        return {"work_order": wo.model_dump(), "decision": "await_approval"}

    def n_approval(s: AgentState) -> AgentState:
        wo = dict(s["work_order"])
        if s.get("decision") == "approved":
            wo.update(status="approved", approver=s.get("approver"))
        elif s.get("decision") == "rejected":
            wo.update(status="rejected", approver=s.get("approver"),
                      reviewer_note=s.get("reviewer_note"))
        return {"work_order": wo}

    g = StateGraph(AgentState)
    for name, fn in [("signals", n_signals), ("vision", n_vision),
                     ("knowledge", n_knowledge), ("fuse", n_fuse), ("approval", n_approval)]:
        g.add_node(name, fn)
    g.set_entry_point("signals")
    g.add_edge("signals", "vision")
    g.add_edge("vision", "knowledge")
    g.add_edge("knowledge", "fuse")
    g.add_conditional_edges("fuse", lambda s: END if s.get("decision") == "declined"
                            else "approval", {"approval": "approval", END: END})
    g.add_edge("approval", END)
    return g.compile(checkpointer=checkpointer or MemorySaver(),
                     interrupt_before=["approval"])
