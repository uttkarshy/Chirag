"""Evaluation Runner for Chirag Model Laboratory (M3.8).

M3.8 EXECUTES CANDIDATES. IT DOES NOT DETERMINE WHICH MODEL IS BETTER.
----------------------------------------------------------------------
The EvaluationRunner is an orchestration component. It runs multiple
EvaluationCandidates against one EvaluationTask and normalizes each
execution into an EvaluationResult.

It does NOT contain quality judgment.
It does NOT score candidates.
It does NOT select a winner.

EXAMPLE SCENARIO:
-----------------
Task:
    "Analyze this investment scenario."

Candidate A:
    model_id = "deepseek/deepseek-chat"
    provider_id = "openrouter"

Candidate B:
    model_id = "meta-llama/llama-3.3-70b-instruct"
    provider_id = "openrouter"

Runner output:
    Candidate A -> EvaluationResult (output, latency, usage, status)
    Candidate B -> EvaluationResult (output, latency, usage, status)

EvaluationSummary:
    results = [Result A, Result B]
    scores = []
    winner_candidate_id = None

Future evaluation logic (M3.10+) will score candidates and determine quality.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..executor import ModelExecutor
from ..schema import ExecutionStatus, ModelExecutionRequest, ModelResult
from ..validation import sanitize_error_message
from .ranking import evaluation_result_from_model_result
from .schema import (
    EvaluationCandidate,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
)
from .validation import (
    InvalidEvaluationCandidateError,
    validate_evaluation_candidate,
    validate_evaluation_summary,
    validate_evaluation_task,
)

_REDACTION_BEARER = re.compile(r'(?i)bearer\s+[a-zA-Z0-9_\-\.]+')
_REDACTION_SK = re.compile(r'(?i)sk-[a-zA-Z0-9_\-\.]{8,}')
_REDACTION_GENERIC = re.compile(
    r'(?i)(api[_-]?key|token|secret|password|auth)[:=\s]+[a-zA-Z0-9_\-\.]{8,}'
)


def _sanitize_error_str(msg: str) -> str:
    """Sanitize error messages and redact sensitive credential patterns."""
    if not msg:
        return "Unknown execution failure."
    sanitized = msg.strip()
    sanitized = _REDACTION_BEARER.sub("Bearer [REDACTED]", sanitized)
    sanitized = _REDACTION_SK.sub("sk-[REDACTED]", sanitized)
    sanitized = _REDACTION_GENERIC.sub(r"\1=[REDACTED]", sanitized)
    return sanitized


class EvaluationRunner:
    """Executes multiple EvaluationCandidates against a single EvaluationTask.

    Orchestrates execution sequentially and deterministically without quality judgment.
    """

    def __init__(self, executor: ModelExecutor) -> None:
        if executor is None or not (
            isinstance(executor, ModelExecutor) or callable(getattr(executor, "execute", None))
        ):
            raise TypeError(
                f"EvaluationRunner requires a ModelExecutor instance, got {type(executor).__name__}."
            )
        self.executor = executor

    def run(
        self,
        task: EvaluationTask,
        candidates: list[EvaluationCandidate],
        metadata: dict[str, str] | None = None,
    ) -> EvaluationSummary:
        """Run candidate models sequentially on the task and return an EvaluationSummary.

        Rules:
        - Candidate order is strictly preserved.
        - Duplicate candidate_ids cause an immediate validation failure before execution begins.
        - One candidate failure or exception does not abort remaining candidates.
        - latency_ms is measured using monotonic clocks and is non-negative.
        - scores is empty and winner_candidate_id is None.
        """
        validate_evaluation_task(task)

        # 1. Validate candidate list and prevent duplicate candidate_ids
        seen_candidate_ids: set[str] = set()
        for candidate in candidates:
            validate_evaluation_candidate(candidate)
            if candidate.candidate_id in seen_candidate_ids:
                raise InvalidEvaluationCandidateError(
                    f"Duplicate candidate_id detected: '{candidate.candidate_id}'. "
                    "Candidate IDs must be unique within an evaluation run."
                )
            seen_candidate_ids.add(candidate.candidate_id)

        # 2. Handle empty candidate list deterministically
        if not candidates:
            summary_meta = {
                "task_id": task.task_id,
                "candidate_count": "0",
                "execution_mode": "evaluation_runner",
            }
            if metadata:
                summary_meta.update(metadata)
            return EvaluationSummary(
                task_id=task.task_id,
                results=[],
                scores=[],
                winner_candidate_id=None,
                metadata=summary_meta,
            )

        # 3. Execute candidates sequentially
        results: list[EvaluationResult] = []
        expected_out = (
            task.expected_output.strip()
            if task.expected_output and task.expected_output.strip()
            else "Valid response for evaluation task."
        )

        for candidate in candidates:
            req_metadata = dict(task.metadata)
            req_metadata.update(candidate.metadata)
            req_metadata["provider_id"] = candidate.provider_id
            req_metadata["task_id"] = task.task_id
            req_metadata["candidate_id"] = candidate.candidate_id

            request = ModelExecutionRequest(
                model_id=candidate.model_id,
                input=task.input,
                input_modalities=task.input_modalities,
                expected_output=expected_out,
                output_modalities=task.output_modalities,
                structured_output=task.structured_output,
                metadata=req_metadata,
            )

            start_time = time.perf_counter()
            try:
                model_result = self.executor.execute(request)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if model_result.error:
                    model_result = model_result.model_copy(
                        update={"error": _sanitize_error_str(model_result.error)}
                    )

                eval_result = evaluation_result_from_model_result(
                    candidate=candidate,
                    model_result=model_result,
                    latency_ms=max(0.0, elapsed_ms),
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                eval_result = EvaluationResult(
                    candidate_id=candidate.candidate_id,
                    model_id=candidate.model_id,
                    provider_id=candidate.provider_id,
                    status=EvaluationStatus.FAILED.value,
                    output=None,
                    error=_sanitize_error_str(sanitize_error_message(exc)),
                    latency_ms=max(0.0, elapsed_ms),
                    usage={},
                    metadata={"error_type": type(exc).__name__},
                )

            results.append(eval_result)

        summary_meta = {
            "task_id": task.task_id,
            "candidate_count": str(len(candidates)),
            "execution_mode": "evaluation_runner",
        }
        if metadata:
            summary_meta.update(metadata)

        summary = EvaluationSummary(
            task_id=task.task_id,
            results=results,
            scores=[],
            winner_candidate_id=None,
            metadata=summary_meta,
        )
        return validate_evaluation_summary(summary)
