from __future__ import annotations

from ..schema import ExecutionStatus, ModelResult
from ..validation import validate_model_result
from .schema import (
    EvaluationCandidate,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
)
from .validation import (
    InvalidEvaluationResultError,
    validate_evaluation_candidate,
    validate_evaluation_result,
    validate_evaluation_score,
    validate_evaluation_summary,
    validate_evaluation_task,
)


def rank_scores(scores: list[EvaluationScore]) -> list[EvaluationScore]:
    """Rank evaluation scores deterministically.

    Ranking ordering:
    1. overall_score descending (higher is better).
    2. candidate_id ascending (alphabetical tie-breaker).

    This ensures 100% deterministic ranking without relying on timestamps,
    randomness, or provider/model bias.
    """
    for score in scores:
        validate_evaluation_score(score)

    return sorted(scores, key=lambda s: (-s.overall_score, s.candidate_id))


def select_winner(
    scores: list[EvaluationScore],
    results: list[EvaluationResult] | None = None,
) -> str | None:
    """Select the winning candidate_id based on scores and execution results.

    Rules:
    - If scores is empty, returns None.
    - If results is provided and all candidates failed, returns None.
    - Failed candidates cannot be selected as the winner.
    - Deterministic tie-breaker is applied via rank_scores.
    """
    if not scores:
        return None

    failed_candidates: set[str] = set()
    if results is not None:
        for res in results:
            validate_evaluation_result(res)
            if res.status == EvaluationStatus.FAILED.value:
                failed_candidates.add(res.candidate_id)

        # If all candidates in results failed, no winner is selected
        if results and all(r.status == EvaluationStatus.FAILED.value for r in results):
            return None

    eligible_scores = [s for s in scores if s.candidate_id not in failed_candidates]
    if not eligible_scores:
        return None

    ranked = rank_scores(eligible_scores)
    return ranked[0].candidate_id


def build_evaluation_summary(
    task: EvaluationTask,
    results: list[EvaluationResult],
    scores: list[EvaluationScore] | None = None,
    metadata: dict[str, str] | None = None,
) -> EvaluationSummary:
    """Aggregate candidate results and scores into a validated EvaluationSummary."""
    validate_evaluation_task(task)
    for res in results:
        validate_evaluation_result(res)

    ranked_scores: list[EvaluationScore] = []
    winner_id: str | None = None

    if scores is not None and len(scores) > 0:
        for sc in scores:
            validate_evaluation_score(sc)
        ranked_scores = rank_scores(scores)
        winner_id = select_winner(ranked_scores, results)

    summary = EvaluationSummary(
        task_id=task.task_id,
        results=results,
        scores=ranked_scores,
        winner_candidate_id=winner_id,
        metadata=metadata or {},
    )
    return validate_evaluation_summary(summary)


def evaluation_result_from_model_result(
    candidate: EvaluationCandidate,
    model_result: ModelResult,
    latency_ms: float = 0.0,
    metadata: dict[str, str] | None = None,
) -> EvaluationResult:
    """Bridge runtime ModelResult into an EvaluationResult for the given candidate.

    Maintains clean architectural decoupling:
    - ModelResult remains strictly an execution-level artifact.
    - EvaluationResult represents the normalized task evaluation record.
    """
    validate_evaluation_candidate(candidate)
    validate_model_result(model_result)

    if candidate.model_id != model_result.model_id:
        raise InvalidEvaluationResultError(
            f"Candidate model_id '{candidate.model_id}' does not match "
            f"ModelResult model_id '{model_result.model_id}'."
        )

    if model_result.status == ExecutionStatus.SUCCESS.value:
        status = EvaluationStatus.SUCCESS.value
        output = model_result.output
        error = None
    else:
        status = EvaluationStatus.FAILED.value
        output = model_result.output
        error = model_result.error or "Execution failed without error message."

    res_metadata = dict(model_result.metadata)
    if metadata:
        res_metadata.update(metadata)

    eval_result = EvaluationResult(
        candidate_id=candidate.candidate_id,
        model_id=candidate.model_id,
        provider_id=candidate.provider_id,
        status=status,
        output=output,
        error=error,
        latency_ms=max(0.0, float(latency_ms)),
        usage=dict(model_result.usage),
        metadata=res_metadata,
    )
    return validate_evaluation_result(eval_result)
