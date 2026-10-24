"""Ported from NB19-NB24 in-notebook asserts."""
import pytest
from aip_agents.evaluation import action_grounded, evidence_consistent, summarize
from aip_agents.graph import WorkOrder, build_graph
from aip_agents.scheduler import Schedule, Technician, WorkOrderSpec, optimize
from aip_agents.tools import query_knowledge, query_signals, query_vision
from aip_core import paths

CORPUS = paths.data / "corpus"
requires_corpus = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")


# ---------- scheduler (verified reference) ----------
DEMO_WOS = [
    WorkOrderSpec("WO1", "pump-3", 4, 3, 0, 24, "hydraulics"),
    WorkOrderSpec("WO2", "conveyor-1", 3, 2, 0, 48, "mechanical"),
    WorkOrderSpec("WO3", "pump-7", 6, 3, 8, 40, "hydraulics"),
    WorkOrderSpec("WO4", "hvac-2", 2, 1, 0, 72, "electrical"),
    WorkOrderSpec("WO5", "press-1", 5, 2, 0, 48, "mechanical"),
    WorkOrderSpec("WO6", "pump-3", 3, 1, 24, 60, "hydraulics"),
    WorkOrderSpec("WO7", "impossible", 4, 2, 0, 10, "welding"),
]
CREW = [Technician("T1", frozenset({"hydraulics", "electrical"}), (0, 48)),
        Technician("T2", frozenset({"mechanical", "hydraulics"}), (8, 56))]


@pytest.fixture(scope="module")
def schedule() -> Schedule:
    return optimize(DEMO_WOS, CREW)


def test_scheduler_optimal(schedule):
    assert schedule.status in ("OPTIMAL", "FEASIBLE")


def test_unskillable_work_order_left_unscheduled(schedule):
    assert schedule.unscheduled == ["WO7"]


def test_skills_and_windows_respected(schedule):
    by_id = {w.id: w for w in DEMO_WOS}
    for a in schedule.assignments:
        w = by_id[a["wo_id"]]
        t = next(t for t in CREW if t.id == a["tech_id"])
        assert w.required_skill in t.skills
        assert a["start_h"] >= max(w.earliest_h, t.shift[0])
        assert a["end_h"] <= min(w.deadline_h, t.shift[1])


def test_no_technician_double_booking(schedule):
    spans: dict[str, list] = {}
    for a in schedule.assignments:
        spans.setdefault(a["tech_id"], []).append((a["start_h"], a["end_h"]))
    for tid, s in spans.items():
        s.sort()
        for (_s1, e1), (s2, _e2) in zip(s, s[1:], strict=False):
            assert s2 >= e1, f"overlap on {tid}"


def test_critical_work_scheduled_early(schedule):
    by_id = {w.id: w for w in DEMO_WOS}
    crit = [a for a in schedule.assignments if by_id[a["wo_id"]].priority == 3]
    assert crit and all(a["start_h"] <= 12 for a in crit)


# ---------- tools ----------
@requires_corpus
def test_query_knowledge_contract():
    r = query_knowledge("bearing envelope BPFO outer race")
    assert r.hits and r.evidence_strength in ("strong", "weak")
    assert r.hits[0].doc_id.startswith("heur-vib")


@requires_corpus
def test_query_knowledge_declines_on_nonsense():
    assert query_knowledge("zzzz qqqq xxxx").evidence_strength == "none"


def test_query_signals_unknown_asset_degrades():
    s = query_signals("definitely-not-an-asset-xyz", db_path="/nonexistent.duckdb")
    assert s.evidence_strength == "none" and s.n_cycles_observed == 0


def test_query_vision_unknown_asset_degrades():
    v = query_vision("nonexistent-asset", findings_path="/nonexistent.json")
    assert v.n_findings == 0 and v.evidence_strength == "none"


# ---------- graph / HITL ----------
@requires_corpus
def test_graph_pauses_before_approval():
    g = build_graph()
    cfg = {"configurable": {"thread_id": "t-pause"}}
    out = g.invoke({"asset_id": "press-1"}, cfg)
    assert out["work_order"]["status"] in ("draft", "declined")
    if out["work_order"]["status"] == "draft":
        assert g.get_state(cfg).next == ("approval",)


@requires_corpus
def test_no_approval_without_human():
    g = build_graph()
    cfg = {"configurable": {"thread_id": "t-noapprove"}}
    out = g.invoke({"asset_id": "press-1"}, cfg)
    assert out["work_order"]["status"] != "approved"


@requires_corpus
def test_declines_on_absent_evidence():
    g = build_graph()
    cfg = {"configurable": {"thread_id": "t-decline"}}
    wo = g.invoke({"asset_id": "ghost-asset-99"}, cfg)["work_order"]
    assert wo["status"] == "declined" and wo["duration_h_est"] == 0


# ---------- evaluation metrics ----------
def _wo(**kw):
    base = dict(wo_id="W", asset_id="a", created_at="t", evidence={"evidence_strength": "weak"},
                recommended_action="do it [heur-vib-0001#2]", required_skill="mechanical",
                duration_h_est=2, priority=1, status="draft")
    base.update(kw)
    return WorkOrder(**base).model_dump()


def test_groundedness_requires_real_chunk():
    valid = {"heur-vib-0001#2"}
    assert action_grounded(_wo(), valid)
    assert not action_grounded(_wo(recommended_action="do it [made-up#9]"), valid)
    assert not action_grounded(_wo(recommended_action="no citation at all"), valid)


def test_declined_orders_need_no_citation():
    wo = _wo(status="declined", priority=1, duration_h_est=0, recommended_action="none")
    assert action_grounded(wo, set()) and evidence_consistent(wo)


def test_priority_must_match_evidence():
    assert not evidence_consistent(_wo(priority=3))
    assert evidence_consistent(_wo(priority=3, evidence={"evidence_strength": "strong"}))


def test_summarize():
    rows = [{"expect_ok": True, "grounded": True, "consistent": True, "status": "draft"},
            {"expect_ok": True, "grounded": True, "consistent": True, "status": "declined"}]
    s = summarize(rows)
    assert s["expectation_match"] == 1.0 and s["declines"] == 1
