from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_contract():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "chirag-gateway",
        "version": "0.2.0",
    }


def test_intent_analyze_forwards_gateway_request(monkeypatch):
    captured = {}

    def fake_call(payload):
        captured.update(payload)
        return {"intent": {"goal": "build an app"}, "questions": []}

    monkeypatch.setattr("app.main._call_intent_engine", fake_call)

    response = client.post(
        "/intent/analyze",
        json={
            "message": "build an app",
            "output": "Android app",
            "requirements": ["offline-first"],
            "constraints": ["no SaaS dependency"],
            "preferences": ["simple UI"],
            "context": ["Chirag"],
            "inputs": [
                {
                    "type": "repository",
                    "id": "uttkarshy/Chirag",
                    "name": "Chirag",
                    "role": "primary_source",
                    "metadata": {"branch": "main"},
                }
            ],
            "clarification_answers": [
                {"question": "What?", "answer": "An Android app"}
            ],
        },
    )

    assert response.status_code == 200
    assert captured == {
        "goal": "build an app",
        "output": "Android app",
        "requirements": ["offline-first"],
        "constraints": ["no SaaS dependency"],
        "preferences": ["simple UI"],
        "context": ["Chirag"],
        "inputs": [
            {
                "type": "repository",
                "id": "uttkarshy/Chirag",
                "name": "Chirag",
                "role": "primary_source",
                "metadata": {"branch": "main"},
            }
        ],
        "clarification_answers": [
            {"question": "What?", "answer": "An Android app"}
        ],
    }
    assert response.json() == {"intent": {"goal": "build an app"}, "questions": []}


def test_chat_returns_needs_clarification(monkeypatch):
    def fake_call(payload):
        return {
            "intent": {
                "goal": payload["goal"],
                "needs_clarification": True,
            },
            "questions": ["What would you like Chirag to produce as the final result?"],
        }

    monkeypatch.setattr("app.main._call_intent_engine", fake_call)

    response = client.post("/chat", json={"message": "help me"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "needs_clarification",
        "intent": {"goal": "help me", "needs_clarification": True},
        "questions": ["What would you like Chirag to produce as the final result?"],
    }


def test_chat_returns_accepted(monkeypatch):
    intent = {
        "goal": "build a landing page",
        "output": "HTML/CSS page",
        "needs_clarification": False,
        "confidence": 0.9,
    }

    monkeypatch.setattr(
        "app.main._call_intent_engine",
        lambda payload: {"intent": intent, "questions": []},
    )

    response = client.post(
        "/chat",
        json={"message": "build a landing page", "output": "HTML/CSS page"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert body["intent"] == intent
    assert "Planner/model routing" in body["message"]


def test_chat_rejects_missing_message():
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_intent_engine_failure_is_503(monkeypatch):
    from fastapi import HTTPException

    def failing_call(payload):
        raise HTTPException(status_code=503, detail="Intent Engine unavailable")

    monkeypatch.setattr("app.main._call_intent_engine", failing_call)

    response = client.post("/chat", json={"message": "build something"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Intent Engine unavailable"}
