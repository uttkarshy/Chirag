"""Structured contracts and schemas for LLM Judge Evaluator (M3.11)."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUPPORTED_JUDGE_DIMENSIONS: frozenset[str] = frozenset(
    {"correctness", "relevance", "completeness", "instruction_following"}
)

DEFAULT_JUDGE_RUBRIC: dict[str, str] = {
    "correctness": "Factual accuracy, truthfulness, and absence of hallucinations or incorrect reasoning.",
    "relevance": "Direct focus on the query without extraneous, off-topic, or distracting material.",
    "completeness": "Thorough coverage of all explicit and implicit sub-tasks, constraints, and questions.",
    "instruction_following": "Strict adherence to formatting, structural requirements, constraints, and length guidelines.",
}


class JudgeCandidateEvaluation(BaseModel):
    """Evaluation output for a single candidate produced by the LLM Judge."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(
        ..., description="Anonymous label of the evaluated candidate (e.g. Candidate A)."
    )
    scores: dict[str, float] = Field(
        ..., description="Mapping of dimension names to scores between 0.0 and 1.0."
    )
    rationale: str = Field(
        ..., description="Reasoning and evidence justifying the assigned scores."
    )
    overall_score: float | None = Field(
        default=None, description="Optional overall score provided by judge."
    )

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("candidate_id cannot be empty.")
        return v.strip()

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("rationale cannot be empty.")
        return v.strip()

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("scores dictionary cannot be empty.")
        for dim, score in v.items():
            if not isinstance(dim, str) or not dim.strip():
                raise ValueError("Dimension name must be a non-empty string.")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"Score for dimension '{dim}' must be numeric.")
            if math.isnan(score):
                raise ValueError(f"Score for dimension '{dim}' cannot be NaN.")
            if math.isinf(score):
                raise ValueError(f"Score for dimension '{dim}' cannot be infinite.")
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Score for dimension '{dim}' must be between 0.0 and 1.0, got {score}."
                )
        return v

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float | None) -> float | None:
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("overall_score must be numeric.")
            if math.isnan(v):
                raise ValueError("overall_score cannot be NaN.")
            if math.isinf(v):
                raise ValueError("overall_score cannot be infinite.")
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"overall_score must be between 0.0 and 1.0, got {v}.")
        return v


class JudgeStructuredOutput(BaseModel):
    """Top-level structured output schema expected from the LLM Judge."""

    model_config = ConfigDict(extra="forbid")

    evaluations: list[JudgeCandidateEvaluation] = Field(
        ..., description="List of candidate evaluations."
    )
