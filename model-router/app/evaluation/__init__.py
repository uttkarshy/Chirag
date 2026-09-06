"""Evaluation contract and ranking package for Chirag Model Laboratory."""

from .ranking import (
    build_evaluation_summary,
    evaluation_result_from_model_result,
    rank_scores,
    select_winner,
)
from .runner import EvaluationRunner
from .schema import (
    EvaluationCandidate,
    EvaluationDimension,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
)
from .validation import (
    EvaluationError,
    InvalidEvaluationCandidateError,
    InvalidEvaluationDimensionError,
    InvalidEvaluationResultError,
    InvalidEvaluationScoreError,
    InvalidEvaluationSummaryError,
    InvalidEvaluationTaskError,
    validate_evaluation_candidate,
    validate_evaluation_dimension,
    validate_evaluation_result,
    validate_evaluation_score,
    validate_evaluation_summary,
    validate_evaluation_task,
)

__all__ = [
    "EvaluationStatus",
    "EvaluationTask",
    "EvaluationCandidate",
    "EvaluationResult",
    "EvaluationDimension",
    "EvaluationScore",
    "EvaluationSummary",
    "EvaluationError",
    "InvalidEvaluationTaskError",
    "InvalidEvaluationCandidateError",
    "InvalidEvaluationResultError",
    "InvalidEvaluationDimensionError",
    "InvalidEvaluationScoreError",
    "InvalidEvaluationSummaryError",
    "validate_evaluation_task",
    "validate_evaluation_candidate",
    "validate_evaluation_result",
    "validate_evaluation_dimension",
    "validate_evaluation_score",
    "validate_evaluation_summary",
    "rank_scores",
    "select_winner",
    "build_evaluation_summary",
    "evaluation_result_from_model_result",
    "EvaluationRunner",
]
