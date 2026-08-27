"""
Database models for the LLM Evaluation Dashboard.
Defines the EvaluationRecord table using SQLModel (Pydantic + SQLAlchemy).
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


class EvaluationRecord(SQLModel, table=True):
    """Stores the result of a single LLM evaluation run."""

    __tablename__ = "evaluations"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_prompt: str
    model_name: str
    prompt_version: str = Field(default="v1")
    output_text: str

    # Quality metrics  (0.0 – 1.0)
    correctness_score: float = Field(default=0.0)
    relevance_score: float = Field(default=0.0)
    faithfulness_score: float = Field(default=0.0)

    # Optional Ground Truth & LLM Judge Feedback
    ground_truth: Optional[str] = Field(default=None)
    judge_reasoning: Optional[str] = Field(default=None)

    # Telemetry & cost
    latency_seconds: float = Field(default=0.0)
    total_tokens: int = Field(default=0)
    estimated_cost_usd: float = Field(default=0.0)
