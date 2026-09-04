import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "0.2.0"
INTENT_ENGINE_URL = os.getenv("CHIRAG_INTENT_ENGINE_URL", "http://intent-engine:8001")
PLANNER_URL = os.getenv("CHIRAG_PLANNER_URL", "http://planner:8002")

app = FastAPI(title="Chirag Gateway", version=APP_VERSION)


class InputSource(BaseModel):
    type: str
    id: str | None = None
    name: str | None = None
    role: str = "supporting_source"
    metadata: dict[str, str] = Field(default_factory=dict)


class ClarificationAnswer(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    message: str
    output: str | None = None
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    inputs: list[InputSource] = Field(default_factory=list)
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list)


def _call_intent_engine(payload: dict) -> dict:
    request = Request(
        f"{INTENT_ENGINE_URL.rstrip('/')}/analyze",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Intent Engine unavailable: {exc}") from exc


def _call_planner(payload: dict) -> dict:
    request = Request(
        f"{PLANNER_URL.rstrip('/')}/plan",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Planner unavailable: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "chirag-gateway", "version": APP_VERSION}


@app.get("/models")
def models() -> dict:
    return {"models": [], "message": "No model providers registered yet."}


@app.post("/intent/analyze")
def analyze_intent(request: ChatRequest) -> dict:
    payload = request.model_dump()
    payload["goal"] = request.message
    payload.pop("message", None)
    return _call_intent_engine(payload)


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    intent_response = _call_intent_engine(
        {
            "goal": request.message,
            "output": request.output,
            "requirements": request.requirements,
            "constraints": request.constraints,
            "preferences": request.preferences,
            "context": request.context,
            "inputs": [item.model_dump() for item in request.inputs],
            "clarification_answers": [answer.model_dump() for answer in request.clarification_answers],
        }
    )

    intent = intent_response["intent"]
    if intent["needs_clarification"]:
        return {
            "status": "needs_clarification",
            "intent": intent,
            "questions": intent_response["questions"],
        }

    planner_response = _call_planner({"intent": intent})
    plan = planner_response["plan"]

    return {
        "status": "accepted",
        "intent": intent,
        "plan": plan,
        "message": "Intent understood and planned. Planner/model routing will execute this plan in a later milestone.",
    }
