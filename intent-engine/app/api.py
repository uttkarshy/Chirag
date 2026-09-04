from fastapi import FastAPI

from .engine import analyze_request
from .schema import AnalyzeRequest, AnalyzeResponse

APP_VERSION = "0.2.0"
app = FastAPI(title="Chirag Intent Engine", version=APP_VERSION)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "chirag-intent-engine", "version": APP_VERSION}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    intent = analyze_request(request)
    return AnalyzeResponse(intent=intent, questions=intent.clarification_questions)
