"""Intelligent Evaluation Framework (M3.10).

M3.10 INTRODUCES EVALUATION OF MODEL OUTPUTS.
----------------------------------------------
Important distinction:
- EvaluationRunner: "Did the model execution produce a result?"
- Evaluator: "How good is that result according to defined evaluation criteria?"

EXAMPLE:
--------
Task:
    "Return the exact value 42."

Candidate A:
    output = "42"

Candidate B:
    output = "41"

Deterministic Evaluator:
    Candidate A: exact_match = 1.0
    Candidate B: exact_match = 0.0

For natural-language tasks without a reference answer, deterministic evaluation
must NOT pretend to understand semantic quality.

FUTURE LLM JUDGE:
------------------
Candidate outputs
      ↓
LLM Judge (Future M3.11+)
      ↓
correctness, relevance, completeness, instruction_following
      ↓
Overall Quality Score
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from .ranking import build_evaluation_summary, rank_scores, select_winner
from .schema import (
    EvaluationDimension,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
)
from .validation import (
    InvalidEvaluationDimensionError,
    UnsupportedEvaluationDimensionError,
    validate_evaluation_dimension,
    validate_evaluation_result,
    validate_evaluation_score,
    validate_evaluation_task,
)

_REDACTION_BEARER = re.compile(r'(?i)bearer\s+[a-zA-Z0-9_\-\.]+')
_REDACTION_SK = re.compile(r'(?i)sk-[a-zA-Z0-9_\-\.]{8,}')
_REDACTION_GENERIC = re.compile(
    r'(?i)(api[_-]?key|token|secret|password|auth)[:=\s]+[a-zA-Z0-9_\-\.]{8,}'
)

SUPPORTED_DETERMINISTIC_DIMENSIONS: frozenset[str] = frozenset(
    {"execution_success", "output_present", "exact_match"}
)


def _sanitize_rationale_text(text: str) -> str:
    """Ensure rationale strings never leak credentials, bearer tokens, or API keys."""
    if not text:
        return ""
    sanitized = text.strip()
    sanitized = _REDACTION_BEARER.sub("Bearer [REDACTED]", sanitized)
    sanitized = _REDACTION_SK.sub("sk-[REDACTED]", sanitized)
    sanitized = _REDACTION_GENERIC.sub(r"\1=[REDACTED]", sanitized)
    return sanitized


class Evaluator(ABC):
    """Abstract base class for provider-neutral model output evaluation."""

    @abstractmethod
    def evaluate(
        self,
        task: EvaluationTask,
        results: list[EvaluationResult],
        dimensions: list[EvaluationDimension],
    ) -> list[EvaluationScore]:
        """Evaluate candidate execution results against specified quality dimensions."""
        pass

    def evaluate_and_summarize(
        self,
        task: EvaluationTask,
        results: list[EvaluationResult],
        dimensions: list[EvaluationDimension],
        metadata: dict[str, str] | None = None,
    ) -> EvaluationSummary:
        """Evaluate candidate execution results and return an aggregated EvaluationSummary."""
        scores = self.evaluate(task, results, dimensions)
        return build_evaluation_summary(
            task=task,
            results=results,
            scores=scores,
            metadata=metadata,
        )


class DeterministicEvaluator(Evaluator):
    """Deterministic evaluator for objective, non-LLM output properties.

    Supported dimensions:
    - 'execution_success': 1.0 if candidate execution status is 'success', else 0.0.
    - 'output_present': 1.0 if candidate output is non-empty string, else 0.0.
    - 'exact_match': 1.0 if candidate output exactly matches task.expected_output (after
      stripping whitespace), else 0.0. Requires task.expected_output to be defined.

    Unsupported dimensions (e.g. semantic similarity, relevance) are explicitly rejected
    to prevent deceptive fake quality scoring without an LLM.
    """

    def evaluate(
        self,
        task: EvaluationTask,
        results: list[EvaluationResult],
        dimensions: list[EvaluationDimension],
    ) -> list[EvaluationScore]:
        """Evaluate candidate results deterministically against supported dimensions."""
        validate_evaluation_task(task)

        if not dimensions:
            raise InvalidEvaluationDimensionError(
                "At least one evaluation dimension must be specified."
            )

        # Validate and inspect requested dimensions
        for dim in dimensions:
            validate_evaluation_dimension(dim)
            if dim.name not in SUPPORTED_DETERMINISTIC_DIMENSIONS:
                raise UnsupportedEvaluationDimensionError(
                    f"Dimension '{dim.name}' is not supported by DeterministicEvaluator. "
                    f"Supported dimensions: {sorted(SUPPORTED_DETERMINISTIC_DIMENSIONS)}"
                )
            if dim.name == "exact_match" and task.expected_output is None:
                raise UnsupportedEvaluationDimensionError(
                    "Dimension 'exact_match' cannot be evaluated because task.expected_output is None."
                )

        if not results:
            return []

        scores: list[EvaluationScore] = []
        total_weight = sum(dim.weight for dim in dimensions)

        for res in results:
            validate_evaluation_result(res)

            dim_scores: dict[str, float] = {}
            for dim in dimensions:
                if dim.name == "execution_success":
                    dim_score = 1.0 if res.status == EvaluationStatus.SUCCESS.value else 0.0
                elif dim.name == "output_present":
                    dim_score = (
                        1.0
                        if (res.output is not None and len(res.output.strip()) > 0)
                        else 0.0
                    )
                elif dim.name == "exact_match":
                    # task.expected_output is verified non-None above
                    expected = task.expected_output.strip() if task.expected_output else ""
                    actual = res.output.strip() if res.output else ""
                    dim_score = 1.0 if (res.output is not None and actual == expected) else 0.0
                else:
                    raise UnsupportedEvaluationDimensionError(
                        f"Unhandled dimension: '{dim.name}'."
                    )
                dim_scores[dim.name] = dim_score

            # Weighted aggregation
            if total_weight <= 0.0:
                overall_score = 0.0
            else:
                weighted_sum = sum(dim_scores[dim.name] * dim.weight for dim in dimensions)
                overall_score = max(0.0, min(1.0, float(weighted_sum / total_weight)))

            # Construct safe rationale
            if res.status == EvaluationStatus.FAILED.value:
                safe_err = _sanitize_rationale_text(res.error or "execution failed")
                rationale = (
                    f"Candidate '{res.candidate_id}' failed execution ({safe_err}). "
                    f"Overall score: {overall_score:.2f}."
                )
            else:
                dim_str = ", ".join(f"{k}={v:.2f}" for k, v in dim_scores.items())
                rationale = (
                    f"Candidate '{res.candidate_id}' evaluated on [{dim_str}]. "
                    f"Overall score: {overall_score:.2f}."
                )

            score = EvaluationScore(
                candidate_id=res.candidate_id,
                overall_score=overall_score,
                dimension_scores=dim_scores,
                rationale=rationale,
                metadata={"evaluator": "DeterministicEvaluator"},
            )
            scores.append(validate_evaluation_score(score))

        return scores
