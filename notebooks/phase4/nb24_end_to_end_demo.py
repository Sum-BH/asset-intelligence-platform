# %% [markdown]
# # NB24 — End-to-End Demo: Alarm → Evidence → Draft → Approval → Optimal Schedule
# The loop-closure demo. Everything below runs on real repo artifacts: DuckDB cycles,
# Phase-3 VisionFindings, the Phase-2 corpus, the LangGraph HITL agent, and CP-SAT.
# This is the demo you record for Write-up #4 and the conference talk.

# %%
import json
import time
from pathlib import Path

DATA = Path("../../data")
t_start = time.time()

_t = Path("nb19_tools.py").read_text().split("# ## 5. Fixture tests")[0]
exec("\n".join(ln for ln in _t.splitlines() if not ln.startswith(("# %%", "# #"))))
_g = Path("nb20_agent_graph.py").read_text().split("# ## 3. Run: draft")[0]
exec("\n".join(ln for ln in _g.splitlines()
               if not ln.startswith(("# %%", "# #", "from __future__"))))
# NB21 defines its own `WorkOrder` (solver input: durations, windows, skills) which is
# a DIFFERENT concept from NB20's business `WorkOrder` (evidence, citations, approval).
# Load the scheduler into its own namespace and alias explicitly — a lesson worth
# keeping at conversion time: `aip_agents.scheduler.WorkOrderSpec` vs `graph.WorkOrder`.
_s = Path("nb21_scheduler.py").read_text().split("# ## 2. Assertion suite")[0]
_sched_ns: dict = {}
exec("\n".join(ln for ln in _s.splitlines()
                if not ln.startswith(("# %%", "# #", "from __future__"))), _sched_ns)
WorkOrderSpec = _sched_ns["WorkOrder"]
Technician = _sched_ns["Technician"]
optimize = _sched_ns["optimize"]

def banner(step, text):
    print(f"\n{'='*72}\n  STEP {step}: {text}\n{'='*72}")

# %% [markdown]
# ## 1. Alarm arrives from the plant

# %%
ALARMS = [{"asset_id": "press-1", "source": "vibration", "raised": "shift-A"},
          {"asset_id": "panel-2", "source": "thermal", "raised": "shift-A"},
          {"asset_id": "taxiway-A3", "source": "camera", "raised": "shift-B"}]
banner(1, "Alarms received")
for a in ALARMS:
    print(f"  ! {a['asset_id']:12s} via {a['source']:10s} ({a['raised']})")

# %% [markdown]
# ## 2. Agent gathers evidence across signals, vision and knowledge

# %%
banner(2, "Agent gathers multimodal evidence (3 tools, HITL graph)")
drafts, cfgs = [], {}
for a in ALARMS:
    cfg = {"configurable": {"thread_id": f"demo24-{a['asset_id']}"}}
    cfgs[a["asset_id"]] = cfg
    out = graph.invoke({"asset_id": a["asset_id"]}, cfg)
    wo, sig = out["work_order"], out["signals"]
    drafts.append(wo)
    print(f"  {a['asset_id']:12s} health={sig['health_index']:.2f} ({sig['trend']:7s}) "
          f"vision={out['vision']['n_findings']} findings → {wo['status']} "
          f"p{wo['priority']} [{wo['required_skill']}]")
    print(f"      action: {wo['recommended_action'][:96]}")

# %% [markdown]
# ## 3. HUMAN GATE — nothing proceeds without approval

# %%
banner(3, "Human approval gate (graph paused before 'approval' node)")
for wo in drafts:
    nxt = graph.get_state(cfgs[wo["asset_id"]]).next
    print(f"  {wo['wo_id']} {wo['asset_id']:12s} status={wo['status']:8s} paused_at={nxt}")
    assert wo["status"] == "draft" and nxt == ("approval",)

DECISIONS = {"press-1": ("approved", None), "panel-2": ("approved", None),
             "taxiway-A3": ("rejected", "FOD already cleared by ops this morning.")}
approved = []
for wo in drafts:
    decision, note = DECISIONS[wo["asset_id"]]
    graph.update_state(cfgs[wo["asset_id"]],
                       {"decision": decision, "approver": "sumedh", "reviewer_note": note})
    final = graph.invoke(None, cfgs[wo["asset_id"]])["work_order"]
    print(f"  → {final['wo_id']} {final['status'].upper()} by {final['approver']}"
          + (f" ({final['reviewer_note']})" if final.get("reviewer_note") else ""))
    if final["status"] == "approved":
        approved.append(final)
assert len(approved) == 2

# %% [markdown]
# ## 4. Solver schedules only what a human approved

# %%
banner(4, "CP-SAT optimal scheduling of approved work orders")
crew = [Technician("T1", frozenset({"hydraulics", "electrical", "airside"}), (0, 24)),
        Technician("T2", frozenset({"mechanical", "hydraulics"}), (0, 24))]
wos = [WorkOrderSpec(w["wo_id"], w["asset_id"], max(int(w["duration_h_est"]), 1),
                     int(w["priority"]), 0, 24, w["required_skill"]) for w in approved]
sch = optimize(wos, crew)
print(f"  solver: {sch.status}  objective={sch.objective:.0f}")
for a in sch.assignments:
    w = next(x for x in approved if x["wo_id"] == a["wo_id"])
    print(f"    {a['wo_id']} {w['asset_id']:12s} → {a['tech_id']} "
          f"[{a['start_h']:>2}h–{a['end_h']:>2}h]  {w['required_skill']}")
assert sch.status in ("OPTIMAL", "FEASIBLE") and len(sch.assignments) == len(approved)

# %% [markdown]
# ## 5. Loop closed — audit trail

# %%
banner(5, "Loop closed: sense → detect → diagnose → prescribe → schedule → act")
elapsed = time.time() - t_start
audit = {"alarms": ALARMS,
         "work_orders": [{"wo_id": w["wo_id"], "asset": w["asset_id"],
                          "status": w["status"], "approver": w.get("approver"),
                          "priority": w["priority"], "citations": w["citations"]}
                         for w in drafts],
         "schedule": sch.assignments, "solver_status": sch.status,
         "human_decisions": {k: v[0] for k, v in DECISIONS.items()},
         "elapsed_seconds": round(elapsed, 1)}
print(f"  {len(ALARMS)} alarms → {len(drafts)} drafts → {len(approved)} approved "
      f"→ {len(sch.assignments)} scheduled")
print(f"  every approval human-gated; every action cites the knowledge base")
print(f"  end-to-end wall time: {elapsed:.1f}s (budget: 180s)")
assert elapsed < 180, "demo must run in under 3 minutes"

json.dump(audit, open(DATA / "processed/nb24_demo_audit.json", "w"), indent=1)
print("\nNB24 complete.")
