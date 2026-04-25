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
    # DriftShield warmup task: prompt-injection, easy difficulty.
    r = tc.post("/reset", json={"task_id": "ds_prompt_injection_access"})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "ds_prompt_injection_access"

    r = tc.post(
        "/step",
        json={
            "action": {
                "assistant_message": "Opening the suspicious admin-grant request.",
                "tool_calls": [{"name": "inbox.open_case", "args": {"case_id": "I-9001"}}],
            }
        },
    )
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert obs["tool_results"][0]["name"] == "inbox.open_case"
    assert "Inbox" in obs["app_summaries"]

    r = tc.get("/state")
    assert r.status_code == 200
    data = r.json()
    assert "episode_id" in data
    assert "step_count" in data
