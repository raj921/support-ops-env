from fastapi.testclient import TestClient
from support_ops_env.server.app import app

tc = TestClient(app)


def test_health_endpoint():
    r = tc.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_schema_endpoint():
    r = tc.get("/schema")
    assert r.status_code == 200
    data = r.json()
    assert "action" in data
    assert "observation" in data
    assert "state" in data


def test_reset_step_and_state_endpoints():
    r = tc.post("/reset", json={"task_id": "easy_vip_sso"})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "easy_vip_sso"

    r = tc.post("/step", json={"action": {"action_type": "view_ticket", "ticket_id": "E-1001"}})
    assert r.status_code == 200
    assert r.json()["observation"]["current_ticket_id"] == "E-1001"

    r = tc.get("/state")
    assert r.status_code == 200
    data = r.json()
    assert "episode_id" in data
    assert "step_count" in data
