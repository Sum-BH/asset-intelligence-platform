"""FastAPI service layer (promoted from NB25).

The API adds transport, never new shapes: every response is a frozen schema from
aip_agents.tools / aip_agents.scheduler.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from aip_agents.scheduler import Technician, WorkOrderSpec, optimize
from aip_agents.tools import query_knowledge, query_signals, query_vision
from aip_core import paths
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel


class ApprovalBody(BaseModel):
    decision: str
    note: str | None = None


class ScheduleBody(BaseModel):
    work_orders: list[dict]
    technicians: list[dict]


def create_app(workorder_dir: str | Path | None = None) -> FastAPI:
    wo_dir = Path(workorder_dir or paths.processed / "workorders")
    app = FastAPI(title="Asset Intelligence Platform", version="1.0.0")

    def require_key(x_api_key: str = Header(default="")) -> str:
        if x_api_key != os.environ.get("AIP_API_KEY", "local-dev-key"):
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return x_api_key

    def require_approver(x_approver: str = Header(default="")) -> str:
        if not x_approver:
            raise HTTPException(status_code=403, detail="X-Approver header required")
        return x_approver

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
        wos = [json.loads(p.read_text()) for p in sorted(wo_dir.glob("*.json"))]
        return [w for w in wos if not status or w["status"] == status]

    @app.patch("/workorders/{wo_id}", dependencies=[Depends(require_key)])
    def r_approve(wo_id: str, body: ApprovalBody, approver: str = Depends(require_approver)):
        p = wo_dir / f"{wo_id}.json"
        if not p.exists():
            raise HTTPException(404, "work order not found")
        if body.decision not in ("approved", "rejected"):
            raise HTTPException(422, "decision must be approved or rejected")
        wo = json.loads(p.read_text())
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

    return app


app = create_app()
