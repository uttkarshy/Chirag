"""Prompt Strategy Optimization Contracts for Chirag (M3.12).

ARCHITECTURAL PRINCIPLES:
-------------------------
MODEL ("What"):
    What intelligence/model is used (e.g. 'deepseek/deepseek-chat').
PROVIDER ("How"):
    Through which service inference is accessed (e.g. 'openrouter', 'local').
COMPUTE ("Where"):
    Where physical or virtual execution runs (AWS, GCP, local GPU).
PROMPT STRATEGY ("How instructed"):
    How the base task is structured, phrased, or framed into a prompt variant.

Prompt strategy optimization operates ABOVE model execution and evaluation.
It does NOT alter model routing or provider selection.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..evaluation.schema import (
    EvaluationCandidate,
    EvaluationDimension,
    EvaluationScore,
    EvaluationSummary,
    EvaluationTask,
)


class PromptStrategy(BaseModel):
    """Specification of a prompt strategy template."""

    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(..., description="Stable, unique identifier for the prompt strategy.")
    name: str = Field(..., description="Human-readable strategy name.")
    description: str = Field(default="", description="Detailed description of the strategy.")
    template: str = Field(
        ...,
        description="Prompt template or instruction framing containing '{task_input}'.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Strategy configuration metadata.",
    )

    @field_validator("strategy_id")
    @classmethod
    def validate_strategy_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("strategy_id must be a non-empty string.")
        return v.strip()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name must be a non-empty string.")
        return v.strip()

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("template must be a non-empty string.")
        return v


class PromptVariant(BaseModel):
    """A rendered prompt variant produced from a base task and a prompt strategy."""

    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(..., description="Deterministic identifier for this rendered variant.")
    strategy_id: str = Field(..., description="Identifier of the strategy that generated this variant.")
    task_id: str = Field(..., description="Identifier of the base task.")
    prompt: str = Field(..., description="The concrete prompt text to be sent to model candidates.")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Variant metadata preserving generation context.",
    )

    @field_validator("variant_id", "strategy_id", "task_id")
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Identifier must be a non-empty string.")
        return v.strip()

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("prompt must be a non-empty string.")
        return v


class StrategyObservation(BaseModel):
    """A structured, factual observation about strategy performance on an evaluation dimension."""

    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(..., description="Identifier of the observed strategy.")
    dimension: str = Field(..., description="Evaluation dimension analyzed.")
    score: float = Field(..., ge=0.0, le=1.0, description="Observed score on this dimension.")
    observation: str = Field(..., description="Factual, deterministic observation text.")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Observation metadata.",
    )

    @field_validator("strategy_id", "dimension")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("score must be between 0.0 and 1.0 inclusive.")
        return v

    @field_validator("observation")
    @classmethod
    def validate_observation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("observation must be a non-empty string.")
        return v.strip()


class StrategyEvaluationScore(BaseModel):
    """Aggregated evaluation score for a prompt strategy across candidate models."""

    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(..., description="Identifier of the evaluated strategy.")
    variant_id: str = Field(..., description="Identifier of the evaluated prompt variant.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Aggregated overall score (0.0 - 1.0).")
    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated dimension scores.",
    )
    candidate_scores: list[EvaluationScore] = Field(
        default_factory=list,
        description="Underlying candidate model evaluation scores.",
    )
    rationale: str | None = Field(default=None, description="Optional explanation or synthesis.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Score metadata.")

    @field_validator("strategy_id", "variant_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Identifier must be a non-empty string.")
        return v.strip()

    @field_validator("overall_score")
    @classmethod
    def validate_overall(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("overall_score must be between 0.0 and 1.0 inclusive.")
        return v

    @field_validator("dimension_scores")
    @classmethod
    def validate_dims(cls, v: dict[str, float]) -> dict[str, float]:
        for dim_name, score in v.items():
            if not isinstance(dim_name, str) or not dim_name.strip():
                raise ValueError("Dimension name must be a non-empty string.")
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Dimension score for '{dim_name}' must be between 0.0 and 1.0 inclusive.")
        return v


class PromptOptimizationExperiment(BaseModel):
    """Specification of a controlled prompt optimization experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(..., description="Unique experiment identifier.")
    task: EvaluationTask = Field(..., description="Base evaluation task (same-task guarantee).")
    candidate_models: list[EvaluationCandidate] = Field(
        ...,
        description="Candidate model/provider configurations executing each prompt variant.",
    )
    candidate_strategies: list[PromptStrategy] = Field(
        ...,
        description="List of prompt strategies to experiment with.",
    )
    evaluation_dimensions: list[EvaluationDimension] = Field(
        ...,
        description="Evaluation dimensions used to score outputs.",
    )
    base_strategy: PromptStrategy | None = Field(
        default=None,
        description="Optional base/control prompt strategy for baseline comparison.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Experiment configuration metadata.",
    )

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("experiment_id cannot be empty.")
        return v.strip()

    @field_validator("candidate_models")
    @classmethod
    def validate_candidates(cls, v: list[EvaluationCandidate]) -> list[EvaluationCandidate]:
        if not v:
            raise ValueError("candidate_models cannot be empty.")
        seen = set()
        for cand in v:
            if cand.candidate_id in seen:
                raise ValueError(f"Duplicate candidate_id '{cand.candidate_id}' detected in experiment.")
            seen.add(cand.candidate_id)
        return v

    @field_validator("candidate_strategies")
    @classmethod
    def validate_strategies(cls, v: list[PromptStrategy]) -> list[PromptStrategy]:
        if not v:
            raise ValueError("candidate_strategies cannot be empty.")
        seen = set()
        for strat in v:
            if strat.strategy_id in seen:
                raise ValueError(f"Duplicate strategy_id '{strat.strategy_id}' detected in experiment.")
            seen.add(strat.strategy_id)
        return v

    @field_validator("evaluation_dimensions")
    @classmethod
    def validate_dimensions(cls, v: list[EvaluationDimension]) -> list[EvaluationDimension]:
        if not v:
            raise ValueError("evaluation_dimensions cannot be empty.")
        seen = set()
        for dim in v:
            if dim.name in seen:
                raise ValueError(f"Duplicate dimension name '{dim.name}' detected in experiment.")
            seen.add(dim.name)
        return v


class PromptOptimizationResult(BaseModel):
    """The outcome of a prompt optimization experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(..., description="Identifier of the experiment.")
    task_id: str = Field(..., description="Identifier of the base task.")
    strategy_scores: list[StrategyEvaluationScore] = Field(
        default_factory=list,
        description="Ranked strategy evaluation scores.",
    )
    winner_strategy_id: str | None = Field(
        default=None,
        description="Identifier of the winning strategy, if one qualified.",
    )
    winner_score: float | None = Field(
        default=None,
        description="Overall score achieved by the winning strategy.",
    )
    winning_strategy: PromptStrategy | None = Field(
        default=None,
        description="The winning prompt strategy object.",
    )
    optimized_prompt: str | None = Field(
        default=None,
        description="The concrete winning prompt text.",
    )
    observations: list[StrategyObservation] = Field(
        default_factory=list,
        description="Structured observations across evaluated strategies.",
    )
    strategy_summaries: dict[str, EvaluationSummary] = Field(
        default_factory=dict,
        description="Per-strategy EvaluationSummary mapping strategy_id to candidate execution/scoring details.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Result metadata.",
    )

    @field_validator("experiment_id", "task_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Identifier must be a non-empty string.")
        return v.strip()

    @field_validator("winner_score")
    @classmethod
    def validate_winner_score(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("winner_score must be between 0.0 and 1.0 inclusive.")
        return v
