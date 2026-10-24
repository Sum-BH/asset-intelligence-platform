"""Ported from NB25-NB28 in-notebook asserts."""
import json

import numpy as np
import pytest
from aip_agents.tools import KnowledgeResult, SignalResult, VisionResult
from aip_core import paths
from aip_serve.api import create_app
from aip_serve.drift import check_batch, psi
from aip_serve.promotion import promote
from fastapi.testclient import TestClient

CORPUS = paths.data / "corpus"
requires_corpus = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AIP_API_KEY", "test-key")
    wo = {"wo_id": "WO-test01", "asset_id": "press-1", "created_at": "t", "evidence": {},
          "recommended_action": "x", "required_skill": "mechanical", "duration_h_est": 2,
          "priority": 1, "status": "draft"}
    (tmp_path / "WO-test01.json").write_text(json.dumps(wo))
    return TestClient(create_app(workorder_dir=tmp_path))


KEY = {"X-API-Key": "test-key"}


def test_health_open(client):
    assert client.get("/health").json()["status"] == "ok"


def test_auth_required(client):
    assert client.get("/signals/press-1/health-index").status_code == 401


@requires_corpus
def test_response_schemas_match_frozen_contracts(client):
    s = client.get("/signals/press-1/health-index", headers=KEY).json()
    assert set(SignalResult.model_fields) == set(s)
    v = client.get("/vision/findings/press-1", headers=KEY).json()
    assert set(VisionResult.model_fields) == set(v)
    k = client.get("/knowledge/query", params={"q": "bearing", "k": 2}, headers=KEY).json()
    assert set(KnowledgeResult.model_fields) == set(k)


def test_approval_requires_approver_identity(client):
    r = client.patch("/workorders/WO-test01", json={"decision": "approved"}, headers=KEY)
    assert r.status_code == 403


def test_approval_records_approver(client):
    r = client.patch("/workorders/WO-test01", json={"decision": "approved", "note": "ok"},
                     headers={**KEY, "X-Approver": "sumedh"})
    assert r.status_code == 200 and r.json()["approver"] == "sumedh"


def test_invalid_decision_rejected(client):
    r = client.patch("/workorders/WO-test01", json={"decision": "maybe"},
                     headers={**KEY, "X-Approver": "sumedh"})
    assert r.status_code == 422


def test_schedule_route(client):
    r = client.post("/schedule/optimize", headers=KEY, json={
        "work_orders": [{"id": "W1", "asset_id": "pump-3", "duration_h": 3, "priority": 3,
                         "earliest_h": 0, "deadline_h": 24, "required_skill": "hydraulics"}],
        "technicians": [{"id": "T1", "skills": ["hydraulics"], "shift": [0, 24]}]})
    body = r.json()
    assert body["status"] in ("OPTIMAL", "FEASIBLE") and body["assignments"]


# ---------- promotion ----------
def test_promotion_blocks_nasa_regression(tmp_path):
    ledger = tmp_path / "ledger.json"
    good = {"model_family": "gb", "nasa_score": 100.0, "rmse": 10.0}
    worse = {"model_family": "rf", "nasa_score": 200.0, "rmse": 12.0}
    assert promote(good, "Staging", ledger)["decision"] == "promoted"
    ev = promote(worse, "Staging", ledger)
    assert ev["decision"] == "refused" and "regression" in ev["why"]


def test_production_requires_reason(tmp_path):
    ledger = tmp_path / "ledger.json"
    cand = {"model_family": "gb", "nasa_score": 100.0}
    assert promote(cand, "Production", ledger)["decision"] == "refused"
    assert promote(cand, "Production", ledger, reason="validated")["decision"] == "promoted"


# ---------- drift ----------
def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, rng.normal(size=2000)) < 0.1


def test_drift_detected_and_no_false_alarm():
    rng = np.random.default_rng(1)
    ref = {"f1": rng.normal(size=2000)}
    stationary = {"f1": rng.normal(size=500)}
    drifted = {"f1": rng.normal(size=500) + 1.0}
    assert check_batch(ref, stationary) == []
    assert any(a["kind"] == "feature_psi" for a in check_batch(ref, drifted))


def test_prediction_drift_detected():
    rng = np.random.default_rng(2)
    ref = {"f1": rng.normal(size=1000)}
    alerts = check_batch(ref, {"f1": rng.normal(size=500)},
                         preds_ref=rng.normal(60, 12, 500), preds_cur=rng.normal(35, 12, 500))
    assert any(a["kind"] == "prediction_ks" for a in alerts)
