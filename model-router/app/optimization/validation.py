"""Validation and domain exceptions for Chirag Prompt Strategy Optimization (M3.12)."""

from __future__ import annotations

import re
from typing import Any

from ..evaluation.validation import (
    validate_evaluation_candidate,
    validate_evaluation_dimension,
    validate_evaluation_task,
)
from .schema import (
    PromptOptimizationExperiment,
    PromptOptimizationResult,
    PromptStrategy,
    PromptVariant,
    StrategyEvaluationScore,
    StrategyObservation,
)

_REDACTION_BEARER = re.compile(r'(?i)bearer\s+[a-zA-Z0-9_\-\.]+')
_REDACTION_SK = re.compile(r'(?i)sk-[a-zA-Z0-9_\-\.]{8,}')
_REDACTION_GENERIC = re.compile(
    r'(?i)(api[_-]?key|token|secret|password|auth)[:=\s]+[a-zA-Z0-9_\-\.]{8,}'
)


def sanitize_text(text: str) -> str:
    """Ensure text never leaks credentials, bearer tokens, or API keys."""
    if not text:
        return ""
    sanitized = text.strip()
    sanitized = _REDACTION_BEARER.sub("Bearer [REDACTED]", sanitized)
    sanitized = _REDACTION_SK.sub("sk-[REDACTED]", sanitized)
    sanitized = _REDACTION_GENERIC.sub(r"\1=[REDACTED]", sanitized)
    return sanitized


class OptimizationError(Exception):
    """Base exception for all prompt strategy optimization errors."""
    pass


class InvalidPromptStrategyError(OptimizationError, ValueError):
    """Raised when a PromptStrategy fails validation or invariant checks."""
    pass


class InvalidPromptVariantError(OptimizationError, ValueError):
    """Raised when a PromptVariant fails validation or invariant checks."""
    pass


class InvalidPromptExperimentError(OptimizationError, ValueError):
    """Raised when a PromptOptimizationExperiment fails validation or invariant checks."""
    pass


class InvalidPromptResultError(OptimizationError, ValueError):
    """Raised when a PromptOptimizationResult fails validation or invariant checks."""
    pass


def validate_prompt_strategy(strategy: PromptStrategy) -> PromptStrategy:
    """Validate that a PromptStrategy adheres to all contract invariants."""
    if not isinstance(strategy, PromptStrategy):
        raise InvalidPromptStrategyError(
            f"Expected PromptStrategy instance, got {type(strategy).__name__}."
        )

    if not strategy.strategy_id or not strategy.strategy_id.strip():
        raise InvalidPromptStrategyError("PromptStrategy strategy_id cannot be empty.")

    if not strategy.name or not strategy.name.strip():
        raise InvalidPromptStrategyError("PromptStrategy name cannot be empty.")

    if not strategy.template or not strategy.template.strip():
        raise InvalidPromptStrategyError("PromptStrategy template cannot be empty.")

    if "{task_input}" not in strategy.template:
        raise InvalidPromptStrategyError(
            "PromptStrategy template must contain '{task_input}' placeholder."
        )

    return strategy


def validate_prompt_variant(variant: PromptVariant) -> PromptVariant:
    """Validate that a PromptVariant adheres to all contract invariants."""
    if not isinstance(variant, PromptVariant):
        raise InvalidPromptVariantError(
            f"Expected PromptVariant instance, got {type(variant).__name__}."
        )

    if not variant.variant_id or not variant.variant_id.strip():
        raise InvalidPromptVariantError("PromptVariant variant_id cannot be empty.")

    if not variant.strategy_id or not variant.strategy_id.strip():
        raise InvalidPromptVariantError("PromptVariant strategy_id cannot be empty.")

    if not variant.task_id or not variant.task_id.strip():
        raise InvalidPromptVariantError("PromptVariant task_id cannot be empty.")

    if not variant.prompt or not variant.prompt.strip():
        raise InvalidPromptVariantError("PromptVariant prompt cannot be empty.")

    return variant


def validate_prompt_experiment(
    experiment: PromptOptimizationExperiment,
) -> PromptOptimizationExperiment:
    """Validate that a PromptOptimizationExperiment adheres to all contract invariants.

    Enforces the Same Task Guarantee:
    All strategies must be tested against the exact same valid EvaluationTask.
    """
    if not isinstance(experiment, PromptOptimizationExperiment):
        raise InvalidPromptExperimentError(
            f"Expected PromptOptimizationExperiment instance, got {type(experiment).__name__}."
        )

    if not experiment.experiment_id or not experiment.experiment_id.strip():
        raise InvalidPromptExperimentError("PromptOptimizationExperiment experiment_id cannot be empty.")

    # Validate base task
    validate_evaluation_task(experiment.task)

    # Validate candidate models
    if not experiment.candidate_models:
        raise InvalidPromptExperimentError("experiment candidate_models cannot be empty.")
    seen_cand_ids: set[str] = set()
    for cand in experiment.candidate_models:
        validate_evaluation_candidate(cand)
        if cand.candidate_id in seen_cand_ids:
            raise InvalidPromptExperimentError(
                f"Duplicate candidate_id '{cand.candidate_id}' in candidate_models."
            )
        seen_cand_ids.add(cand.candidate_id)

    # Validate candidate strategies
    if not experiment.candidate_strategies:
        raise InvalidPromptExperimentError("experiment candidate_strategies cannot be empty.")
    seen_strat_ids: set[str] = set()
    for strat in experiment.candidate_strategies:
        validate_prompt_strategy(strat)
        if strat.strategy_id in seen_strat_ids:
            raise InvalidPromptExperimentError(
                f"Duplicate strategy_id '{strat.strategy_id}' in candidate_strategies."
            )
        seen_strat_ids.add(strat.strategy_id)

    # Validate base strategy if present
    if experiment.base_strategy is not None:
        validate_prompt_strategy(experiment.base_strategy)

    # Validate evaluation dimensions
    if not experiment.evaluation_dimensions:
        raise InvalidPromptExperimentError("experiment evaluation_dimensions cannot be empty.")
    seen_dims: set[str] = set()
    for dim in experiment.evaluation_dimensions:
        validate_evaluation_dimension(dim)
        if dim.name in seen_dims:
            raise InvalidPromptExperimentError(
                f"Duplicate evaluation dimension '{dim.name}' in experiment."
            )
        seen_dims.add(dim.name)

    return experiment


def validate_prompt_result(result: PromptOptimizationResult) -> PromptOptimizationResult:
    """Validate that a PromptOptimizationResult adheres to all contract invariants."""
    if not isinstance(result, PromptOptimizationResult):
        raise InvalidPromptResultError(
            f"Expected PromptOptimizationResult instance, got {type(result).__name__}."
        )

    if not result.experiment_id or not result.experiment_id.strip():
        raise InvalidPromptResultError("PromptOptimizationResult experiment_id cannot be empty.")

    if not result.task_id or not result.task_id.strip():
        raise InvalidPromptResultError("PromptOptimizationResult task_id cannot be empty.")

    if result.winner_score is not None and not (0.0 <= result.winner_score <= 1.0):
        raise InvalidPromptResultError(
            f"winner_score must be between 0.0 and 1.0, got {result.winner_score}."
        )

    return result
