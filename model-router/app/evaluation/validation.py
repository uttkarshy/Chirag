from __future__ import annotations

from typing import Any

from ..schema import Modality
from .schema import (
    EvaluationCandidate,
    EvaluationDimension,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
)


class EvaluationError(Exception):
    """Base domain exception for evaluation operations."""
    pass


class InvalidEvaluationTaskError(EvaluationError, ValueError):
    """Raised when an EvaluationTask violates contract invariants."""
    pass


class InvalidEvaluationCandidateError(EvaluationError, ValueError):
    """Raised when an EvaluationCandidate violates contract invariants."""
    pass


class InvalidEvaluationResultError(EvaluationError, ValueError):
    """Raised when an EvaluationResult violates contract invariants."""
    pass


class InvalidEvaluationDimensionError(EvaluationError, ValueError):
    """Raised when an EvaluationDimension violates contract invariants."""
    pass


class InvalidEvaluationScoreError(EvaluationError, ValueError):
    """Raised when an EvaluationScore violates contract invariants."""
    pass


class InvalidEvaluationSummaryError(EvaluationError, ValueError):
    """Raised when an EvaluationSummary violates contract invariants."""
    pass


def validate_evaluation_task(task: EvaluationTask) -> EvaluationTask:
    """Validate that an EvaluationTask adheres to contract invariants."""
    if not isinstance(task, EvaluationTask):
        raise InvalidEvaluationTaskError(
            f"Expected EvaluationTask instance, got {type(task).__name__}."
        )
    if not task.task_id or not task.task_id.strip():
        raise InvalidEvaluationTaskError("task_id cannot be empty.")
    if not task.input or not task.input.strip():
        raise InvalidEvaluationTaskError("input cannot be empty.")
    if not task.input_modalities:
        raise InvalidEvaluationTaskError("input_modalities cannot be empty.")
    if not task.output_modalities:
        raise InvalidEvaluationTaskError("output_modalities cannot be empty.")

    allowed = {m.value for m in Modality}
    for m in task.input_modalities:
        if m not in allowed:
            raise InvalidEvaluationTaskError(f"Invalid input modality '{m}'.")
    for m in task.output_modalities:
        if m not in allowed:
            raise InvalidEvaluationTaskError(f"Invalid output modality '{m}'.")

    return task


def validate_evaluation_candidate(candidate: EvaluationCandidate) -> EvaluationCandidate:
    """Validate that an EvaluationCandidate adheres to contract invariants."""
    if not isinstance(candidate, EvaluationCandidate):
        raise InvalidEvaluationCandidateError(
            f"Expected EvaluationCandidate instance, got {type(candidate).__name__}."
        )
    if not candidate.candidate_id or not candidate.candidate_id.strip():
        raise InvalidEvaluationCandidateError("candidate_id cannot be empty.")
    if not candidate.model_id or not candidate.model_id.strip():
        raise InvalidEvaluationCandidateError("model_id cannot be empty.")
    if not candidate.provider_id or not candidate.provider_id.strip():
        raise InvalidEvaluationCandidateError("provider_id cannot be empty.")
    return candidate


def validate_evaluation_result(result: EvaluationResult) -> EvaluationResult:
    """Validate that an EvaluationResult adheres to contract invariants."""
    if not isinstance(result, EvaluationResult):
        raise InvalidEvaluationResultError(
            f"Expected EvaluationResult instance, got {type(result).__name__}."
        )
    if not result.candidate_id or not result.candidate_id.strip():
        raise InvalidEvaluationResultError("candidate_id cannot be empty.")
    if not result.model_id or not result.model_id.strip():
        raise InvalidEvaluationResultError("model_id cannot be empty.")
    if not result.provider_id or not result.provider_id.strip():
        raise InvalidEvaluationResultError("provider_id cannot be empty.")

    allowed_statuses = {s.value for s in EvaluationStatus}
    if result.status not in allowed_statuses:
        raise InvalidEvaluationResultError(
            f"Invalid status '{result.status}'. Allowed: {sorted(allowed_statuses)}"
        )

    if result.status == EvaluationStatus.SUCCESS.value and result.output is None:
        raise InvalidEvaluationResultError("Successful EvaluationResult must contain an output.")
    if result.status == EvaluationStatus.FAILED.value and not (result.error and result.error.strip()):
        raise InvalidEvaluationResultError("Failed EvaluationResult must contain an error description.")

    if result.latency_ms < 0.0:
        raise InvalidEvaluationResultError("latency_ms cannot be negative.")

    return result


def validate_evaluation_dimension(dimension: EvaluationDimension) -> EvaluationDimension:
    """Validate that an EvaluationDimension adheres to contract invariants."""
    if not isinstance(dimension, EvaluationDimension):
        raise InvalidEvaluationDimensionError(
            f"Expected EvaluationDimension instance, got {type(dimension).__name__}."
        )
    if not dimension.name or not dimension.name.strip():
        raise InvalidEvaluationDimensionError("Dimension name cannot be empty.")
    if dimension.weight < 0.0:
        raise InvalidEvaluationDimensionError("Dimension weight cannot be negative.")
    return dimension


def validate_evaluation_score(score: EvaluationScore) -> EvaluationScore:
    """Validate that an EvaluationScore adheres to contract invariants."""
    if not isinstance(score, EvaluationScore):
        raise InvalidEvaluationScoreError(
            f"Expected EvaluationScore instance, got {type(score).__name__}."
        )
    if not score.candidate_id or not score.candidate_id.strip():
        raise InvalidEvaluationScoreError("candidate_id cannot be empty.")
    if not (0.0 <= score.overall_score <= 1.0):
        raise InvalidEvaluationScoreError(
            f"overall_score must be between 0.0 and 1.0, got {score.overall_score}."
        )
    for dim_name, dim_score in score.dimension_scores.items():
        if not dim_name or not dim_name.strip():
            raise InvalidEvaluationScoreError("Dimension name cannot be empty in dimension_scores.")
        if not (0.0 <= dim_score <= 1.0):
            raise InvalidEvaluationScoreError(
                f"Dimension score for '{dim_name}' must be between 0.0 and 1.0, got {dim_score}."
            )
    return score


def validate_evaluation_summary(summary: EvaluationSummary) -> EvaluationSummary:
    """Validate that an EvaluationSummary adheres to contract invariants."""
    if not isinstance(summary, EvaluationSummary):
        raise InvalidEvaluationSummaryError(
            f"Expected EvaluationSummary instance, got {type(summary).__name__}."
        )
    if not summary.task_id or not summary.task_id.strip():
        raise InvalidEvaluationSummaryError("task_id cannot be empty.")

    for res in summary.results:
        validate_evaluation_result(res)

    for sc in summary.scores:
        validate_evaluation_score(sc)

    if summary.winner_candidate_id is not None:
        if not summary.winner_candidate_id.strip():
            raise InvalidEvaluationSummaryError("winner_candidate_id cannot be blank.")
        # If results are provided, verify the winner did not have a failed status
        result_map = {r.candidate_id: r for r in summary.results}
        if summary.winner_candidate_id in result_map:
            winner_res = result_map[summary.winner_candidate_id]
            if winner_res.status == EvaluationStatus.FAILED.value:
                raise InvalidEvaluationSummaryError(
                    f"Candidate '{summary.winner_candidate_id}' failed execution and cannot be declared winner."
                )

    return summary
