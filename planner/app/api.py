from fastapi import FastAPI, HTTPException
from .engine import create_plan
from .schema import PlanRequest, PlanResponse

APP_VERSION = "0.1.0"
app = FastAPI(title="Chirag Planner", version=APP_VERSION)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "chirag-planner", "version": APP_VERSION}


@app.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest) -> PlanResponse:
    try:
        execution_plan = create_plan(request.intent)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PlanResponse(plan=execution_plan)
