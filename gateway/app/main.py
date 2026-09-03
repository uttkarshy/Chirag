from fastapi import FastAPI
from pydantic import BaseModel

APP_VERSION = "0.1.0"

app = FastAPI(title="Chirag Gateway", version=APP_VERSION)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "chirag-gateway",
        "version": APP_VERSION,
    }


@app.get("/models")
def models() -> dict:
    return {
        "models": [],
        "message": "No model providers registered yet.",
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    return {
        "status": "accepted",
        "message": "Gateway is alive. Model routing will be connected in a later milestone.",
        "input": request.message,
    }
