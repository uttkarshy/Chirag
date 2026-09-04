from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "chirag-planner",
        "version": "0.1.0",
    }


def test_plan_success():
    response = client.post(
        "/plan",
        json={
            "intent": {
                "goal": "Summarize research paper",
                "output": "Executive overview",
                "inputs": [
                    {"type": "file", "name": "paper.pdf", "role": "primary_source"}
                ],
                "requirements": ["Include methodology highlights"],
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "plan" in data
    plan = data["plan"]
    assert plan["goal"] == "Summarize research paper"
    assert plan["desired_output"] == "Executive overview"
    assert len(plan["steps"]) == 3  # inspect_source, generation, verification
    assert plan["requires_tools"] is False


def test_plan_validation_failure_needs_clarification():
    response = client.post(
        "/plan",
        json={
            "intent": {
                "goal": "Do something",
                "output": "result",
                "needs_clarification": True,
                "missing_information": ["clearer goal"],
            }
        },
    )
    assert response.status_code == 422
    assert "clarification" in response.json()["detail"].lower()


def test_plan_validation_failure_empty_goal():
    response = client.post(
        "/plan",
        json={
            "intent": {
                "goal": "   ",
                "output": "result",
            }
        },
    )
    assert response.status_code == 422
    assert "empty goal" in response.json()["detail"].lower()
