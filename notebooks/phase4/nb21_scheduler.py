# %% [markdown]
# # NB21 — CP-SAT Maintenance Schedule Optimizer (R29)
# The prescriptive capstone: approved work orders + technicians (skills, shifts) →
# provably optimal assignment and start times. **The LLM drafts, the solver decides,
# the human approved** — scheduling is a constraint problem, so an LLM guessing
# schedules would undermine the explainability thesis.
# Consumes NB20's approved work orders; emits schedule JSON for NB24.
# Conversion target: `aip_agents/scheduler.py`.

# %%
import json
from pathlib import Path

DATA = Path("../../data")

# %% [markdown]
# ## 1. Model (verified: OPTIMAL + full assertion suite, ortools 9.15)

# %%
"""(cell body) Maintenance-schedule optimizer core — see notebook header.

Given approved work orders, technicians with skills, and shift windows,
produce an optimal assignment + start time per work order that:
  - respects technician skills and shift windows,
  - never double-books a technician (NoOverlap),
  - schedules within each work order's [earliest, deadline],
  - minimizes total weighted delay (priority-weighted start times)
    plus a large penalty for any work order left unscheduled.

I/O contract (frozen for Phase 4 §7.2):
  WorkOrder: id, asset_id, duration_h, priority (1=low..3=critical),
             earliest_h, deadline_h, required_skill
  Technician: id, skills (set), shift (start_h, end_h)
  Result: list of {wo_id, tech_id, start_h, end_h} + unscheduled ids
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

UNSCHEDULED_PENALTY = 10_000


@dataclass(frozen=True)
class WorkOrder:
    id: str
    asset_id: str
    duration_h: int
    priority: int
    earliest_h: int
    deadline_h: int
    required_skill: str


@dataclass(frozen=True)
class Technician:
    id: str
    skills: frozenset[str]
    shift: tuple[int, int]  # (start_h, end_h) within horizon


@dataclass
class Schedule:
    assignments: list[dict] = field(default_factory=list)
    unscheduled: list[str] = field(default_factory=list)
    objective: float = 0.0
    status: str = ""


def optimize(work_orders: list[WorkOrder], technicians: list[Technician],
             horizon_h: int = 168, time_limit_s: float = 10.0) -> Schedule:
    m = cp_model.CpModel()

    # per (wo, tech): optional interval + presence literal
    presence, starts, intervals = {}, {}, {}
    for wo in work_orders:
        for t in technicians:
            if wo.required_skill not in t.skills:
                continue
            lo = max(wo.earliest_h, t.shift[0])
            hi = min(wo.deadline_h, t.shift[1]) - wo.duration_h
            if hi < lo:
                continue
            key = (wo.id, t.id)
            presence[key] = m.new_bool_var(f"p_{wo.id}_{t.id}")
            starts[key] = m.new_int_var(lo, hi, f"s_{wo.id}_{t.id}")
            intervals[key] = m.new_optional_fixed_size_interval_var(
                starts[key], wo.duration_h, presence[key], f"i_{wo.id}_{t.id}")

    # each WO assigned to at most one tech; track scheduled flag
    scheduled = {}
    for wo in work_orders:
        lits = [presence[k] for k in presence if k[0] == wo.id]
        scheduled[wo.id] = m.new_bool_var(f"sched_{wo.id}")
        if lits:
            m.add(sum(lits) == 1).only_enforce_if(scheduled[wo.id])
            m.add(sum(lits) == 0).only_enforce_if(scheduled[wo.id].negated())
        else:
            m.add(scheduled[wo.id] == 0)

    # no technician double-booking
    for t in technicians:
        ivs = [intervals[k] for k in intervals if k[1] == t.id]
        if ivs:
            m.add_no_overlap(ivs)

    # objective: priority-weighted starts + unscheduled penalty
    terms = []
    by_id = {w.id: w for w in work_orders}
    for (wo_id, _tid), s in starts.items():
        w = by_id[wo_id]
        terms.append(w.priority * s)
    for wo in work_orders:
        terms.append(UNSCHEDULED_PENALTY * wo.priority
                     * (1 - scheduled[wo.id]))
    m.minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.solve(m)

    out = Schedule(status=solver.status_name(status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out.unscheduled = [w.id for w in work_orders]
        return out
    out.objective = solver.objective_value
    for wo in work_orders:
        placed = False
        for key, lit in presence.items():
            if key[0] == wo.id and solver.value(lit):
                st = solver.value(starts[key])
                out.assignments.append({"wo_id": wo.id, "tech_id": key[1],
                                        "start_h": st,
                                        "end_h": st + wo.duration_h})
                placed = True
        if not placed:
            out.unscheduled.append(wo.id)
    out.assignments.sort(key=lambda a: a["start_h"])
    return out



# %% [markdown]
# ## 2. Assertion suite — the exit gate (skills, shifts, deadlines, no double-booking)

# %%
wos = [
    WorkOrder("WO1", "pump-3", 4, 3, 0, 24, "hydraulics"),
    WorkOrder("WO2", "conveyor-1", 3, 2, 0, 48, "mechanical"),
    WorkOrder("WO3", "pump-7", 6, 3, 8, 40, "hydraulics"),
    WorkOrder("WO4", "hvac-2", 2, 1, 0, 72, "electrical"),
    WorkOrder("WO5", "press-1", 5, 2, 0, 48, "mechanical"),
    WorkOrder("WO6", "pump-3", 3, 1, 24, 60, "hydraulics"),
    WorkOrder("WO7", "impossible", 4, 2, 0, 10, "welding"),   # no technician has the skill
]
techs = [
    Technician("T1", frozenset({"hydraulics", "electrical"}), (0, 48)),
    Technician("T2", frozenset({"mechanical", "hydraulics"}), (8, 56)),
]
sch = optimize(wos, techs)
print("status:", sch.status, "objective:", sch.objective)
for a in sch.assignments:
    print(f"  {a['wo_id']}: {a['tech_id']}  [{a['start_h']:>2}h -> {a['end_h']:>2}h]")
print("unscheduled:", sch.unscheduled)

assert sch.status in ("OPTIMAL", "FEASIBLE")
assert sch.unscheduled == ["WO7"], "only the unskillable WO may be unscheduled"
by_id = {w.id: w for w in wos}
by_tech = {}
for a in sch.assignments:
    w, t = by_id[a["wo_id"]], next(t for t in techs if t.id == a["tech_id"])
    assert w.required_skill in t.skills
    assert a["start_h"] >= max(w.earliest_h, t.shift[0])
    assert a["end_h"] <= min(w.deadline_h, t.shift[1])
    by_tech.setdefault(a["tech_id"], []).append((a["start_h"], a["end_h"]))
for tid, spans in by_tech.items():
    spans.sort()
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert s2 >= e1, f"overlap on {tid}"
assert all(a["start_h"] <= 12 for a in sch.assignments if by_id[a["wo_id"]].priority == 3)
print("ALL SCHEDULER ASSERTIONS PASSED")

# %% [markdown]
# ## 3. Schedule the *real* approved work orders from NB20

# %%
wo_dir = DATA / "processed/workorders"
approved = [json.loads(p.read_text()) for p in sorted(wo_dir.glob("*.json"))]
approved = [w for w in approved if w["status"] == "approved"]
print(f"{len(approved)} approved work orders from NB20")

real_wos = [WorkOrder(w["wo_id"], w["asset_id"], max(int(w["duration_h_est"]), 1),
                      int(w["priority"]), 0, 72, w["required_skill"]) for w in approved]
crew = [Technician("T1", frozenset({"hydraulics", "electrical", "airside"}), (0, 48)),
        Technician("T2", frozenset({"mechanical", "hydraulics"}), (0, 48))]
real_sched = optimize(real_wos, crew) if real_wos else Schedule(status="NO_INPUT")
for a in real_sched.assignments:
    print(f"  {a['wo_id']} -> {a['tech_id']} [{a['start_h']}h-{a['end_h']}h]")
assert not real_wos or real_sched.status in ("OPTIMAL", "FEASIBLE")

# %% [markdown]
# ## 4. Scale check — 50 work orders under a 10 s solver budget
# Honest finding: at 50 WOs / 6 technicians CP-SAT returns a FEASIBLE schedule that
# places every work order, but does **not prove optimality** within 10 s. That is the
# right trade for an interactive planner (a good schedule now beats a proven-optimal
# one later); raise `time_limit_s` for overnight planning runs if a proof is wanted.
# The gate therefore checks: every WO placed, and the call returns inside its budget.

# %%
import random, time
random.seed(4)
skills = ["hydraulics", "mechanical", "electrical"]
big = [WorkOrder(f"B{i}", f"asset-{i%17}", random.randint(1, 6), random.randint(1, 3),
                 0, random.choice([48, 72, 120]), random.choice(skills)) for i in range(50)]
crew_big = [Technician(f"T{j}", frozenset(random.sample(skills, 2)), (0, 120)) for j in range(6)]
t0 = time.time()
big_sched = optimize(big, crew_big, horizon_h=120, time_limit_s=10)
elapsed = time.time() - t0
print(f"50 WOs / 6 techs: {big_sched.status} in {elapsed:.2f}s, "
      f"{len(big_sched.assignments)} scheduled, {len(big_sched.unscheduled)} unscheduled")
print("optimality proven:" , big_sched.status == "OPTIMAL")
assert big_sched.status in ("OPTIMAL", "FEASIBLE"), "no feasible schedule found"
assert not big_sched.unscheduled, "every schedulable WO must be placed"
assert elapsed < 12, "solver must return within its budget (10s limit + build time)"

json.dump({"demo": {"status": sch.status, "assignments": sch.assignments,
                    "unscheduled": sch.unscheduled},
           "real": {"status": real_sched.status, "assignments": real_sched.assignments},
           "scale": {"n": 50, "seconds": round(elapsed, 2), "status": big_sched.status,
                     "optimality_proven": big_sched.status == "OPTIMAL",
                     "all_placed": not big_sched.unscheduled}},
          open(DATA / "processed/nb21_schedule.json", "w"), indent=1)
print("NB21 complete.")
