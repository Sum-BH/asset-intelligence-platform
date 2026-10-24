# %% [markdown]
# # NB20 — LangGraph Work-Order Agent with Human-in-the-Loop (R27, R28, R32)
# Real LangGraph state machine with checkpointing and `interrupt_before` the approval
# node — no path reaches `approved` without a human. Planner backend switch: local LLM
# (Ollama, guarded) ⇄ deterministic evidence-fusion planner (sandbox). Declining on
# weak evidence is a first-class outcome, not an error.
# Conversion target: `aip_agents/graph.py`.

# %%
import json
import uuid
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Literal, TypedDict

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

DATA = Path("../../data")

# tools from NB19 (file handoff: re-declare by executing its definitions only)
_src = Path("nb19_tools.py").read_text().split("# ## 5. Fixture tests")[0]
exec("\n".join(ln for ln in _src.splitlines() if not ln.startswith(("# %%", "# #"))))

# %% [markdown]
# ## 1. WorkOrder schema (frozen — Phase 5 API serves this unchanged)

# %%
class WorkOrder(BaseModel):
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

SKILL_MAP = {"press-1": "mechanical", "pump-3": "hydraulics", "pump-7": "hydraulics",
             "panel-2": "electrical", "taxiway-A3": "airside", "unit-3": "mechanical"}

# %% [markdown]
# ## 2. Graph nodes — gather → fuse → (interrupt) → approve/reject

# %%
def n_signals(s: AgentState) -> AgentState:
    return {"signals": query_signals(s["asset_id"]).model_dump()}


def n_vision(s: AgentState) -> AgentState:
    return {"vision": query_vision(s["asset_id"]).model_dump()}


def n_knowledge(s: AgentState) -> AgentState:
    sig = s.get("signals", {})
    q = (f"{'rising' if sig.get('trend') == 'rising' else ''} health index degradation "
         f"action window RUL planning")
    return {"knowledge": query_knowledge(q).model_dump()}


def _plan_llm(sig, vis, kno) -> dict | None:
    try:
        requests.get("http://localhost:11434", timeout=2)
    except Exception:
        return None
    prompt = ("Given the tool outputs, reply ONLY as JSON with keys action (<=25 words, "
              "cite [chunk_id]), priority (1-3), duration_h (int). Use only provided facts.\n"
              f"SIGNALS:{json.dumps(sig)}\nVISION:{json.dumps({'n': vis['n_findings']})}\n"
              f"KNOWLEDGE:{json.dumps([h['chunk_id'] for h in kno['hits']])}")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "qwen2.5:14b-instruct-q4_K_M", "prompt": prompt,
                            "stream": False, "options": {"temperature": 0.1}}, timeout=180)
    import re as _re
    m = _re.search(r"\{.*\}", r.json()["response"], _re.S)
    return json.loads(m.group(0)) if m else None


def _plan_rules(sig, vis, kno) -> dict:
    cite = kno["hits"][0]["chunk_id"] if kno["hits"] else "none"
    pri = 3 if (sig["evidence_strength"] == "strong" and vis["n_findings"] > 0) else (
        2 if sig["evidence_strength"] == "strong" else 1)
    hrs = 6 if pri == 3 else (4 if pri == 2 else 2)
    act = (f"Inspect {sig['asset_id']}: health index {sig['health_index']:.2f} "
           f"({sig['trend']}), RUL ~{sig['rul_estimate_cycles']:.0f} cycles; "
           f"schedule before RUL minus RMSE [{cite}]")
    return {"action": act, "priority": pri, "duration_h": hrs}


def n_fuse(s: AgentState) -> AgentState:
    sig, vis, kno = s["signals"], s["vision"], s["knowledge"]
    # R32: decline honestly when evidence is insufficient
    if sig["evidence_strength"] == "none" and vis["evidence_strength"] == "none":
        wo = WorkOrder(wo_id="WO-" + uuid.uuid4().hex[:6], asset_id=s["asset_id"],
                       created_at=datetime.now(UTC).isoformat(),
                       evidence=sig, findings=[], citations=[],
                       recommended_action="Insufficient evidence: no signal or vision "
                                          "support. No work order raised.",
                       required_skill=SKILL_MAP.get(s["asset_id"], "mechanical"),
                       duration_h_est=0, priority=1, status="declined")
        return {"work_order": wo.model_dump(), "decision": "declined"}
    plan = _plan_llm(sig, vis, kno) or _plan_rules(sig, vis, kno)
    wo = WorkOrder(wo_id="WO-" + uuid.uuid4().hex[:6], asset_id=s["asset_id"],
                   created_at=datetime.now(UTC).isoformat(),
                   evidence=sig, findings=vis["findings"][:3],
                   citations=[h["chunk_id"] for h in kno["hits"]],
                   recommended_action=plan["action"], priority=int(plan["priority"]),
                   duration_h_est=int(plan["duration_h"]),
                   required_skill=SKILL_MAP.get(s["asset_id"], "mechanical"),
                   status="draft")
    return {"work_order": wo.model_dump(), "decision": "await_approval"}


def n_approval(s: AgentState) -> AgentState:
    """Runs only AFTER a human resumes the graph with a decision in state."""
    wo = dict(s["work_order"])
    if s.get("decision") == "approved":
        wo.update(status="approved", approver=s.get("approver"))
    elif s.get("decision") == "rejected":
        wo.update(status="rejected", approver=s.get("approver"),
                  reviewer_note=s.get("reviewer_note"))
    return {"work_order": wo}


def route(s: AgentState) -> str:
    return END if s.get("decision") == "declined" else "approval"

g = StateGraph(AgentState)
for name, fn in [("signals", n_signals), ("vision", n_vision),
                 ("knowledge", n_knowledge), ("fuse", n_fuse), ("approval", n_approval)]:
    g.add_node(name, fn)
g.set_entry_point("signals")
g.add_edge("signals", "vision")
g.add_edge("vision", "knowledge")
g.add_edge("knowledge", "fuse")
g.add_conditional_edges("fuse", route, {"approval": "approval", END: END})
g.add_edge("approval", END)
graph = g.compile(checkpointer=MemorySaver(), interrupt_before=["approval"])
print("graph nodes:", list(graph.get_graph().nodes))

# %% [markdown]
# ## 3. Run: draft → INTERRUPT → human approves → resume

# %%
cfg = {"configurable": {"thread_id": "demo-press-1"}}
state = graph.invoke({"asset_id": "press-1"}, cfg)
snap = graph.get_state(cfg)
draft = state["work_order"]
print("draft:", draft["wo_id"], "| status:", draft["status"], "| next:", snap.next)
assert draft["status"] == "draft" and snap.next == ("approval",), "must pause before approval"

graph.update_state(cfg, {"decision": "approved", "approver": "sumedh"})
final = graph.invoke(None, cfg)["work_order"]
print("after human approval:", final["status"], "by", final["approver"])
assert final["status"] == "approved" and final["approver"] == "sumedh"

# %% [markdown]
# ## 4. Rejection path and decline path (R32)

# %%
cfg2 = {"configurable": {"thread_id": "demo-reject"}}
graph.invoke({"asset_id": "panel-2"}, cfg2)
graph.update_state(cfg2, {"decision": "rejected", "approver": "sumedh",
                          "reviewer_note": "Covered by this week's PM."})
rej = graph.invoke(None, cfg2)["work_order"]
assert rej["status"] == "rejected" and rej["reviewer_note"]
print("rejected path OK:", rej["reviewer_note"])

cfg3 = {"configurable": {"thread_id": "demo-decline"}}
dec = graph.invoke({"asset_id": "ghost-asset-99"}, cfg3)["work_order"]
assert dec["status"] == "declined" and graph.get_state(cfg3).next == ()
print("declined path OK:", dec["recommended_action"][:60])

out = DATA / "processed/workorders"; out.mkdir(exist_ok=True)
for wo in (final, rej, dec):
    (out / f"{wo['wo_id']}.json").write_text(json.dumps(wo, indent=1))
print(f"saved {len(list(out.glob('*.json')))} work orders")
print("NB20 complete.")
