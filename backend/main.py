"""
FastAPI application – REST API for the LLM Evaluation Dashboard.

Endpoints
---------
POST /evaluate   – Submit a prompt for parallel LLM evaluation.
GET  /results    – Retrieve stored evaluation records.
GET  /health     – Simple liveness check.
"""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.database import get_session, init_db
from backend.evaluator import AVAILABLE_MODELS, run_evaluation
from backend.models import EvaluationRecord

app = FastAPI(
    title="LLM Evaluation Dashboard API",
    description="Evaluate, compare, and monitor outputs from multiple LLM models.",
    version="1.0.0",
)

# Allow Streamlit (or any frontend) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup() -> None:
    init_db()


# ── Request / Response schemas ───────────────────────────────────────────────
class EvaluationRequest(BaseModel):
    prompt: str
    models: list[str] = AVAILABLE_MODELS
    prompt_version: str = "v1"
    ground_truth: str | None = None


class EvaluationResponse(BaseModel):
    id: str
    model_name: str
    input_prompt: str
    prompt_version: str
    correctness_score: float
    relevance_score: float
    faithfulness_score: float
    ground_truth: str | None = None
    judge_reasoning: str | None = None
    latency_seconds: float
    total_tokens: int
    estimated_cost_usd: float
    output_text: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/evaluate", response_model=list[EvaluationResponse])
async def evaluate(
    request: EvaluationRequest,
    session: Session = Depends(get_session),
):
    """Submit a prompt and evaluate it against the selected models in parallel."""
    records = await run_evaluation(
        prompt=request.prompt,
        models=request.models,
        prompt_version=request.prompt_version,
        ground_truth=request.ground_truth,
    )
    for record in records:
        session.add(record)
    session.commit()

    # Refresh so that default values (id, timestamp) are populated
    for record in records:
        session.refresh(record)

    return records


@app.get("/results", response_model=list[dict])
def get_results(
    limit: int = 100,
    session: Session = Depends(get_session),
):
    """Return the most recent evaluation records."""
    statement = (
        select(EvaluationRecord)
        .order_by(EvaluationRecord.timestamp.desc())  # type: ignore[arg-type]
        .limit(limit)
    )
    records = session.exec(statement).all()
    return [record.model_dump() for record in records]


@app.get("/health")
def health():
    return {"status": "ok"}
