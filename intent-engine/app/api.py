from fastapi import FastAPI, HTTPException

from .engine import analyze_request
from .schema import AnalyzeRequest, AnalyzeResponse
from .validation import validate_intent

APP_VERSION = "0.2.0"
app = FastAPI(title="Chirag Intent Engine", version=APP_VERSION)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "chirag-intent-engine", "version": APP_VERSION}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    intent = analyze_request(request)
    try:
        intent = validate_intent(intent)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AnalyzeResponse(intent=intent, questions=intent.clarification_questions)
