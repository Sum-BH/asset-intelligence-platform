# %% [markdown]
# # NB25 — FastAPI Service Layer over All Modules (R33, R34)
# The API adds **transport, never new shapes**: every response is a frozen schema from
# Phases 1–4. Auth v1: API key header; approval endpoints additionally require an
# approver identity (audit field). Verified in-process with FastAPI's TestClient —
# every route exercised, contract equality asserted against the tool types.
# Conversion target: `aip_serve/api.py`.

# %%
import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

DATA = Path("../../data")
os.environ.setdefault("AIP_API_KEY", "local-dev-key")

# tools + scheduler (file handoff from Phase 4)
_t = Path("../phase4/nb19_tools.py").read_text().split("# ## 5. Fixture tests")[0]
exec("\n".join(ln for ln in _t.splitlines() if not ln.startswith(("# %%", "# #"))))
_s = Path("../phase4/nb21_scheduler.py").read_text().split("# ## 2. Assertion suite")[0]
_sched: dict = {}
exec("\n".join(ln for ln in _s.splitlines()
                if not ln.startswith(("# %%", "# #", "from __future__"))), _sched)
WorkOrderSpec, Technician, optimize = _sched["WorkOrder"], _sched["Technician"], _sched["optimize"]

WO_DIR = DATA / "processed/workorders"

# %% [markdown]
# ## 1. Auth dependencies (R34)

# %%
def require_key(x_api_key: str = Header(default="")) -> str:
    if x_api_key != os.environ["AIP_API_KEY"]:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return x_api_key


def require_approver(x_approver: str = Header(default="")) -> str:
    if not x_approver:
        raise HTTPException(status_code=403, detail="X-Approver header required")
    return x_approver

# %% [markdown]
# ## 2. Routes — one per platform capability

# %%
app = FastAPI(title="Asset Intelligence Platform", version="0.5.0")


class ApprovalBody(BaseModel):
    decision: str
    note: str | None = None


class ScheduleBody(BaseModel):
    work_orders: list[dict]
    technicians: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/signals/{asset_id}/health-index", dependencies=[Depends(require_key)])
def r_signals(asset_id: str, window: int = 30):
    return query_signals(asset_id, window).model_dump()


@app.get("/rul/{asset_id}", dependencies=[Depends(require_key)])
def r_rul(asset_id: str):
    s = query_signals(asset_id)
    return {"asset_id": asset_id, "rul_estimate_cycles": s.rul_estimate_cycles,
            "model": s.rul_model, "rmse_cycles": s.rul_rmse_cycles,
            "evidence_strength": s.evidence_strength}


@app.get("/vision/findings/{asset_id}", dependencies=[Depends(require_key)])
def r_vision(asset_id: str):
    return query_vision(asset_id).model_dump()


@app.get("/knowledge/query", dependencies=[Depends(require_key)])
def r_knowledge(q: str, k: int = 3):
    return query_knowledge(q, k).model_dump()


@app.get("/workorders", dependencies=[Depends(require_key)])
def r_workorders(status: str | None = None):
    wos = [json.loads(p.read_text()) for p in sorted(WO_DIR.glob("*.json"))]
    return [w for w in wos if not status or w["status"] == status]


@app.patch("/workorders/{wo_id}", dependencies=[Depends(require_key)])
def r_approve(wo_id: str, body: ApprovalBody, approver: str = Depends(require_approver)):
    p = WO_DIR / f"{wo_id}.json"
    if not p.exists():
        raise HTTPException(404, "work order not found")
    wo = json.loads(p.read_text())
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(422, "decision must be approved or rejected")
    wo.update(status=body.decision, approver=approver, reviewer_note=body.note)
    p.write_text(json.dumps(wo, indent=1))
    return wo


@app.post("/schedule/optimize", dependencies=[Depends(require_key)])
def r_schedule(body: ScheduleBody):
    wos = [WorkOrderSpec(**w) for w in body.work_orders]
    techs = [Technician(t["id"], frozenset(t["skills"]), tuple(t["shift"]))
             for t in body.technicians]
    sch = optimize(wos, techs)
    return {"status": sch.status, "assignments": sch.assignments,
            "unscheduled": sch.unscheduled, "objective": sch.objective}

# %% [markdown]
# ## 3. Contract tests — every route, auth, and schema equality

# %%
client = TestClient(app)
KEY = {"X-API-Key": os.environ["AIP_API_KEY"]}

assert client.get("/health").json()["status"] == "ok"
assert client.get("/signals/press-1/health-index").status_code == 401     # no key
r = client.get("/signals/press-1/health-index", headers=KEY)
assert r.status_code == 200
body = r.json()
assert set(SignalResult.model_fields) == set(body), "response must equal frozen schema"
print("signals   :", body["health_index"], body["trend"], body["evidence_strength"])

v = client.get("/vision/findings/press-1", headers=KEY).json()
assert set(VisionResult.model_fields) == set(v)
k = client.get("/knowledge/query", params={"q": "bearing envelope BPFO", "k": 2},
               headers=KEY).json()
assert set(KnowledgeResult.model_fields) == set(k) and k["hits"]
print("vision    :", v["n_findings"], "findings | knowledge:", [h["chunk_id"] for h in k["hits"]])

rul = client.get("/rul/press-1", headers=KEY).json()
assert rul["model"] and rul["rmse_cycles"] > 0
print("rul       :", rul["rul_estimate_cycles"], "cycles via", rul["model"])

wos = client.get("/workorders", headers=KEY).json()
print("workorders:", len(wos), "on file")
assert isinstance(wos, list) and wos

# approval requires BOTH key and approver identity
target = next(w["wo_id"] for w in wos)
assert client.patch(f"/workorders/{target}", json={"decision": "approved"},
                    headers=KEY).status_code == 403
ok = client.patch(f"/workorders/{target}", json={"decision": "approved", "note": "via API"},
                  headers={**KEY, "X-Approver": "sumedh"})
assert ok.status_code == 200 and ok.json()["approver"] == "sumedh"
assert client.patch(f"/workorders/{target}", json={"decision": "maybe"},
                    headers={**KEY, "X-Approver": "sumedh"}).status_code == 422
print("approval  : key+approver enforced, invalid decision rejected")

sch = client.post("/schedule/optimize", headers=KEY, json={
    "work_orders": [{"id": "W1", "asset_id": "pump-3", "duration_h": 3, "priority": 3,
                     "earliest_h": 0, "deadline_h": 24, "required_skill": "hydraulics"}],
    "technicians": [{"id": "T1", "skills": ["hydraulics"], "shift": [0, 24]}]}).json()
assert sch["status"] in ("OPTIMAL", "FEASIBLE") and sch["assignments"]
print("schedule  :", sch["status"], sch["assignments"])

spec = client.get("/openapi.json").json()
routes = sorted(spec["paths"])
print("openapi   :", len(routes), "paths")
assert "/health" in routes and "/schedule/optimize" in routes

json.dump({"routes": routes, "auth": "X-API-Key + X-Approver on approvals",
           "contract_equality": "verified against SignalResult/VisionResult/KnowledgeResult"},
          open(DATA / "processed/nb25_api.json", "w"), indent=1)
print("NB25 complete.")
