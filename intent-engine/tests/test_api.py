from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_complete_request():
    response = client.post(
        "/analyze",
        json={
            "goal": "Summarize the supplied report",
            "output": "a concise summary",
            "inputs": [
                {"type": "file", "name": "report.pdf", "role": "primary_source"}
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"]["needs_clarification"] is False
    assert body["questions"] == []


def test_analyze_with_clarification_answer():
    response = client.post(
        "/analyze",
        json={
            "goal": "Analyze this report",
            "clarification_answers": [
                {
                    "question": "What would you like Chirag to produce as the final result?",
                    "answer": "A risk summary",
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"]["output"] == "A risk summary"
    assert body["intent"]["needs_clarification"] is False
