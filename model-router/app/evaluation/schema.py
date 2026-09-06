"""Evaluation Contracts for Chirag Model Laboratory (M3.7).

ARCHITECTURAL SEPARATION:
-------------------------
MODEL ("What"):
    The logical intelligence being evaluated
    (e.g., 'deepseek/deepseek-chat', 'meta-llama/llama-3.3-70b-instruct').

PROVIDER ("How"):
    The inference service / protocol adapter through which the model execution was accessed
    (e.g., 'openrouter', 'local', 'nim').

EVALUATION ("Quality & Performance Assessment"):
    The neutral contracts and scoring criteria comparing multiple candidate executions against the same task.

COMPUTE ("Where"):
    The physical or virtual infrastructure location where execution occurs.
    Not evaluated directly in M3.7; reserved for future Compute Router decisions.

EXAMPLE COMPARISON SCENARIO:
----------------------------
Task: "Analyze this investment scenario."

Candidate A:
    candidate_id = "cand-1"
    model_id = "deepseek/deepseek-chat"
    provider_id = "openrouter"

Candidate B:
    candidate_id = "cand-2"
    model_id = "meta-llama/llama-3.3-70b-instruct"
    provider_id = "openrouter"

Candidate C (Future):
    candidate_id = "cand-3"
    model_id = "deepseek/deepseek-chat"
    provider_id = "nim"

All three can participate as distinct EvaluationCandidates for the same EvaluationTask.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schema import Modality, ModelResult


class EvaluationStatus(str, Enum):
    """Allowed status values for an evaluation candidate execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvaluationTask(BaseModel):
    """A benchmark task against which multiple model/provider candidates are evaluated."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., description="Unique identifier of the evaluation task.")
    input: str = Field(..., description="The user prompt or input content to be evaluated.")
    expected_output: str | None = Field(
        default=None,
        description="Optional reference or expected target output.",
    )
    input_modalities: list[str] = Field(
        default_factory=lambda: [Modality.TEXT.value],
        description="Accepted input modalities.",
    )
    output_modalities: list[str] = Field(
        default_factory=lambda: [Modality.TEXT.value],
        description="Produced output modalities.",
    )
    structured_output: bool = Field(
        default=False,
        description="Whether output is expected to conform to structured schema.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Task metadata.",
    )

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("task_id must be a non-empty string.")
        return v.strip()

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("input must be a non-empty string.")
        return v

    @field_validator("input_modalities", "output_modalities")
    @classmethod
    def validate_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Modalities list cannot be empty.")
        allowed = {m.value for m in Modality}
        seen = set()
        normalized = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Modality item must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed:
                raise ValueError(f"Invalid modality '{item}'. Allowed: {sorted(allowed)}")
            if norm in seen:
                raise ValueError(f"Duplicate modality detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)
        return sorted(normalized)


class EvaluationCandidate(BaseModel):
    """Represents a specific model + provider configuration being evaluated on a task."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., description="Unique candidate identifier within an evaluation run.")
    model_id: str = Field(..., description="Logical model identifier.")
    provider_id: str = Field(..., description="Provider adapter identifier.")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Candidate-specific configuration metadata.",
    )

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        return v.strip()

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_id must be a non-empty string.")
        return v.strip()

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("provider_id must be a non-empty string.")
        return v.strip()


class EvaluationResult(BaseModel):
    """The normalized execution result of a candidate on an EvaluationTask."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., description="Identifier of the candidate evaluated.")
    model_id: str = Field(..., description="Logical model identifier.")
    provider_id: str = Field(..., description="Provider adapter identifier.")
    status: str = Field(..., description="Evaluation status: success, failed, or skipped.")
    output: str | None = Field(default=None, description="Produced output content if successful.")
    error: str | None = Field(default=None, description="Error message if execution failed.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Round-trip execution latency in milliseconds.")
    usage: dict[str, int] = Field(default_factory=dict, description="Observed usage metrics (tokens, etc.).")
    metadata: dict[str, str] = Field(default_factory=dict, description="Result metadata.")

    @field_validator("candidate_id", "model_id", "provider_id")
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Identifier must be a non-empty string.")
        return v.strip()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {s.value for s in EvaluationStatus}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid evaluation status '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("latency_ms cannot be negative.")
        return v


class EvaluationDimension(BaseModel):
    """Definition of an evaluation quality dimension."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the evaluation dimension (e.g. correctness, relevance, factuality).")
    description: str = Field(default="", description="Detailed criteria description for this dimension.")
    weight: float = Field(default=1.0, ge=0.0, description="Relative weight of this dimension (must be non-negative).")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Dimension name must be a non-empty string.")
        return v.strip()

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Dimension weight must be non-negative (>= 0.0).")
        return v


class EvaluationScore(BaseModel):
    """The scored evaluation of a candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., description="Identifier of the candidate evaluated.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0.")
    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension scores between 0.0 and 1.0.",
    )
    rationale: str | None = Field(default=None, description="Optional explanation or critique.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Score metadata.")

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        return v.strip()

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("overall_score must be between 0.0 and 1.0 inclusive.")
        return v

    @field_validator("dimension_scores")
    @classmethod
    def validate_dimension_scores(cls, v: dict[str, float]) -> dict[str, float]:
        for dim_name, score in v.items():
            if not isinstance(dim_name, str) or not dim_name.strip():
                raise ValueError("Dimension score name must be a non-empty string.")
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Dimension score for '{dim_name}' must be between 0.0 and 1.0 inclusive, got {score}."
                )
        return v


class EvaluationSummary(BaseModel):
    """Container representing the aggregated outcome of evaluating multiple candidates on a task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., description="Identifier of the task evaluated.")
    results: list[EvaluationResult] = Field(
        default_factory=list,
        description="Execution results of all candidates.",
    )
    scores: list[EvaluationScore] = Field(
        default_factory=list,
        description="Evaluation scores assigned to candidates.",
    )
    winner_candidate_id: str | None = Field(
        default=None,
        description="Candidate ID of the top-ranked candidate, if determined.",
    )
    metadata: dict[str, str] = Field(default_factory=dict, description="Summary metadata.")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("task_id must be a non-empty string.")
        return v.strip()
