"""CP-SAT maintenance schedule optimizer (promoted from NB21).

ADR-003: the solver's input type is `WorkOrderSpec`, distinct from the business
`WorkOrder` in aip_agents.graph. They were both called WorkOrder in the notebooks
and collided at runtime in NB24; the rename is permanent.


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
class WorkOrderSpec:
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


def optimize(work_orders: list[WorkOrderSpec], technicians: list[Technician],
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




# Backwards-compatible alias for notebook code being migrated.
WorkOrder = WorkOrderSpec
