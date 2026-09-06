"""LLM Judge Evaluator Implementation (M3.11).

Consumes existing EvaluationTask and EvaluationResult objects and semantically
evaluates model outputs using an injected ModelExecutor without quality-faking
or provider bias.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ..executor import ModelExecutor
from ..schema import ExecutionStatus, Modality, ModelExecutionRequest, ModelResult
from ..validation import sanitize_error_message
from .evaluator import Evaluator, _sanitize_rationale_text
from .judge_prompt import JudgePromptBuilder, get_anonymous_label
from .judge_schema import (
    DEFAULT_JUDGE_RUBRIC,
    SUPPORTED_JUDGE_DIMENSIONS,
    JudgeStructuredOutput,
)
from .schema import (
    EvaluationDimension,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationTask,
)
from .validation import (
    InvalidEvaluationDimensionError,
    InvalidJudgeResponseError,
    JudgeEvaluationError,
    JudgeExecutionError,
    UnsupportedEvaluationDimensionError,
    validate_evaluation_dimension,
    validate_evaluation_result,
    validate_evaluation_score,
    validate_evaluation_task,
)

_ALLOWED_ITEM_FIELDS = frozenset(
    {"candidate_id", "candidate_label", "scores", "rationale", "overall_score"}
)


def _extract_json_text(text: str) -> str:
    """Strip markdown code fence blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_and_validate_judge_response(
    raw_output: str,
    expected_labels: list[str],
    requested_dimensions: list[str],
) -> dict[str, dict[str, Any]]:
    """Parse raw judge output and validate against expected candidates and dimensions.

    Returns a mapping from candidate_label -> {
        "scores": dict[str, float],
        "rationale": str,
        "overall_score": float | None,
    }
    """
    if not raw_output or not raw_output.strip():
        raise InvalidJudgeResponseError("Judge model returned empty output.")

    cleaned_text = _extract_json_text(raw_output)

    try:
        data = json.loads(cleaned_text)
    except Exception as exc:
        raise InvalidJudgeResponseError(f"Judge returned malformed JSON: {exc}") from exc

    # Unpack candidate evaluations list
    if isinstance(data, dict):
        if "evaluations" in data:
            eval_list = data["evaluations"]
        elif "candidates" in data:
            eval_list = data["candidates"]
        elif "results" in data:
            eval_list = data["results"]
        elif "candidate_id" in data or "candidate_label" in data:
            eval_list = [data]
        else:
            raise InvalidJudgeResponseError(
                "Judge JSON output must contain an 'evaluations' list."
            )
    elif isinstance(data, list):
        eval_list = data
    else:
        raise InvalidJudgeResponseError(
            f"Expected JSON object or list from judge, got {type(data).__name__}."
        )

    if not isinstance(eval_list, list):
        raise InvalidJudgeResponseError("Evaluations container must be a list.")

    parsed: dict[str, dict[str, Any]] = {}
    seen_labels: set[str] = set()

    for idx, item in enumerate(eval_list):
        if not isinstance(item, dict):
            raise InvalidJudgeResponseError(
                f"Evaluation item at index {idx} must be a dictionary, got {type(item).__name__}."
            )

        # Check for unexpected extra fields
        for key in item:
            if key not in _ALLOWED_ITEM_FIELDS:
                raise InvalidJudgeResponseError(
                    f"Unexpected field '{key}' in candidate evaluation at index {idx}."
                )

        # Extract label/id
        raw_label = item.get("candidate_id") or item.get("candidate_label")
        if not raw_label or not isinstance(raw_label, str) or not raw_label.strip():
            raise InvalidJudgeResponseError(
                f"Evaluation item at index {idx} must contain a non-empty string candidate identifier."
            )
        label = raw_label.strip()

        # Check unknown candidate
        if label not in expected_labels:
            raise InvalidJudgeResponseError(
                f"Unknown candidate '{label}' in judge response. Expected one of: {expected_labels}."
            )

        # Check duplicate candidate evaluation
        if label in seen_labels:
            raise InvalidJudgeResponseError(
                f"Duplicate evaluation detected for candidate '{label}'."
            )
        seen_labels.add(label)

        # Check scores dictionary
        if "scores" not in item:
            raise InvalidJudgeResponseError(
                f"Missing 'scores' field for candidate '{label}'."
            )
        scores_dict = item["scores"]
        if not isinstance(scores_dict, dict):
            raise InvalidJudgeResponseError(
                f"'scores' field for candidate '{label}' must be a dictionary."
            )

        # Check for missing requested dimensions
        for req_dim in requested_dimensions:
            if req_dim not in scores_dict:
                raise InvalidJudgeResponseError(
                    f"Missing score for requested dimension '{req_dim}' for candidate '{label}'."
                )

        # Check for unknown dimensions
        for dim_name in scores_dict:
            if dim_name not in requested_dimensions:
                raise InvalidJudgeResponseError(
                    f"Unknown dimension '{dim_name}' in judge response for candidate '{label}'. "
                    f"Requested dimensions: {requested_dimensions}."
                )

        # Validate numeric bounds
        cleaned_scores: dict[str, float] = {}
        for dim_name, val in scores_dict.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise InvalidJudgeResponseError(
                    f"Score for dimension '{dim_name}' on candidate '{label}' must be numeric, got {type(val).__name__}."
                )
            if math.isnan(val):
                raise InvalidJudgeResponseError(
                    f"Score for dimension '{dim_name}' on candidate '{label}' cannot be NaN."
                )
            if math.isinf(val):
                raise InvalidJudgeResponseError(
                    f"Score for dimension '{dim_name}' on candidate '{label}' cannot be infinite."
                )
            if not (0.0 <= val <= 1.0):
                raise InvalidJudgeResponseError(
                    f"Score for dimension '{dim_name}' on candidate '{label}' must be between 0.0 and 1.0, got {val}."
                )
            cleaned_scores[dim_name] = float(val)

        # Check rationale
        if "rationale" not in item:
            raise InvalidJudgeResponseError(
                f"Missing 'rationale' for candidate '{label}'."
            )
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise InvalidJudgeResponseError(
                f"Candidate '{label}' evaluation must contain a non-empty rationale string."
            )

        parsed[label] = {
            "scores": cleaned_scores,
            "rationale": rationale.strip(),
            "overall_score": float(item["overall_score"]) if item.get("overall_score") is not None else None,
        }

    # Verify all expected candidates were evaluated
    for expected_label in expected_labels:
        if expected_label not in seen_labels:
            raise InvalidJudgeResponseError(
                f"Missing evaluation for candidate '{expected_label}' in judge response."
            )

    return parsed


class LLMJudgeEvaluator(Evaluator):
    """Semantic model output evaluator powered by an LLM Judge.

    Architecture:
    - Provider-neutral: relies on an injected ModelExecutor rather than direct provider calls.
    - Blind evaluation: candidate model_id and provider_id are strictly excluded from the prompt.
    - Strict validation: malformed JSON, missing dimensions, NaN, or out-of-bounds scores fail safely.
    - Deterministic ranking: outputs standard EvaluationScore objects compatible with M3.10 ranking.
    """

    def __init__(
        self,
        executor: ModelExecutor,
        judge_model_id: str,
        judge_provider_id: str | None = None,
        supported_dimensions: list[str] | set[str] | None = None,
        rubric: dict[str, str] | None = None,
        prompt_builder: JudgePromptBuilder | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        if executor is None or not (
            isinstance(executor, ModelExecutor)
            or callable(getattr(executor, "execute", None))
        ):
            raise TypeError(
                f"LLMJudgeEvaluator requires a ModelExecutor instance, got {type(executor).__name__}."
            )
        if not judge_model_id or not judge_model_id.strip():
            raise ValueError("judge_model_id must be a non-empty string.")

        self.executor = executor
        self.judge_model_id = judge_model_id.strip()
        self.judge_provider_id = judge_provider_id.strip() if judge_provider_id else None
        self.supported_dimensions = (
            frozenset(supported_dimensions)
            if supported_dimensions is not None
            else SUPPORTED_JUDGE_DIMENSIONS
        )
        self.rubric = dict(rubric) if rubric else dict(DEFAULT_JUDGE_RUBRIC)
        self.prompt_builder = prompt_builder or JudgePromptBuilder(default_rubric=self.rubric)
        self.metadata = dict(metadata) if metadata else {}

    def evaluate(
        self,
        task: EvaluationTask,
        results: list[EvaluationResult],
        dimensions: list[EvaluationDimension],
    ) -> list[EvaluationScore]:
        """Evaluate candidate execution results semantically using the judge model.

        Guarantees:
        - All candidate results are evaluated against all requested dimensions.
        - Blind evaluation: candidate model/provider identities are hidden from the judge.
        - One candidate failure does not prevent evaluating other candidates.
        - Result scores strictly preserve original candidate order.
        """
        validate_evaluation_task(task)

        if not dimensions:
            raise InvalidEvaluationDimensionError(
                "At least one evaluation dimension must be specified."
            )

        for dim in dimensions:
            validate_evaluation_dimension(dim)
            if dim.name not in self.supported_dimensions:
                raise UnsupportedEvaluationDimensionError(
                    f"Dimension '{dim.name}' is not supported by LLMJudgeEvaluator. "
                    f"Supported dimensions: {sorted(self.supported_dimensions)}"
                )

        if not results:
            return []

        for res in results:
            validate_evaluation_result(res)

        # Partition into candidates eligible for judge evaluation vs failed candidates
        candidates_to_judge: list[EvaluationResult] = []
        failed_results: list[EvaluationResult] = []

        for res in results:
            if res.status == EvaluationStatus.FAILED.value or res.output is None:
                failed_results.append(res)
            else:
                candidates_to_judge.append(res)

        scores_by_candidate_id: dict[str, EvaluationScore] = {}

        # Handle failed candidates directly without invoking the judge
        for res in failed_results:
            dim_scores = {dim.name: 0.0 for dim in dimensions}
            safe_err = _sanitize_rationale_text(res.error or "Execution failed.")
            score_meta = {
                "evaluator": "LLMJudgeEvaluator",
                "judge_model_id": self.judge_model_id,
            }
            if self.judge_provider_id:
                score_meta["judge_provider_id"] = self.judge_provider_id

            score = EvaluationScore(
                candidate_id=res.candidate_id,
                overall_score=0.0,
                dimension_scores=dim_scores,
                rationale=f"Candidate execution failed ({safe_err}). Overall score: 0.00.",
                metadata=score_meta,
            )
            scores_by_candidate_id[res.candidate_id] = validate_evaluation_score(score)

        # If candidates produced outputs, invoke judge model
        if candidates_to_judge:
            label_to_candidate_id: dict[str, str] = {}
            anonymous_candidates: list[dict[str, str]] = []

            for i, res in enumerate(candidates_to_judge):
                label = get_anonymous_label(i)
                label_to_candidate_id[label] = res.candidate_id
                anonymous_candidates.append({
                    "label": label,
                    "output": res.output or "",
                })

            prompt_text = self.prompt_builder.build_prompt(
                task=task,
                anonymous_candidates=anonymous_candidates,
                dimensions=dimensions,
                custom_rubric=self.rubric,
            )

            req_meta = dict(self.metadata)
            if self.judge_provider_id:
                req_meta["provider_id"] = self.judge_provider_id
            req_meta["evaluation_task_id"] = task.task_id
            req_meta["evaluation_role"] = "llm_judge"

            judge_request = ModelExecutionRequest(
                model_id=self.judge_model_id,
                input=prompt_text,
                input_modalities=[Modality.TEXT.value],
                expected_output="Strict JSON containing evaluation scores and rationale.",
                output_modalities=[Modality.TEXT.value],
                structured_output=True,
                metadata=req_meta,
            )

            try:
                judge_result = self.executor.execute(judge_request)
            except Exception as exc:
                raise JudgeExecutionError(
                    f"Judge model execution raised an exception: {_sanitize_rationale_text(str(exc))}"
                ) from exc

            if judge_result.status == ExecutionStatus.FAILED.value:
                raise JudgeExecutionError(
                    f"Judge model execution failed: {_sanitize_rationale_text(judge_result.error or 'Unknown failure')}"
                )

            if not judge_result.output or not judge_result.output.strip():
                raise InvalidJudgeResponseError("Judge model returned empty output.")

            parsed_evaluations = _parse_and_validate_judge_response(
                raw_output=judge_result.output,
                expected_labels=list(label_to_candidate_id.keys()),
                requested_dimensions=[dim.name for dim in dimensions],
            )

            total_weight = sum(dim.weight for dim in dimensions)

            for label, eval_data in parsed_evaluations.items():
                cid = label_to_candidate_id[label]
                dim_scores = {dim.name: float(eval_data["scores"][dim.name]) for dim in dimensions}

                if total_weight <= 0.0:
                    overall_score = 0.0
                else:
                    weighted_sum = sum(dim_scores[dim.name] * dim.weight for dim in dimensions)
                    overall_score = max(0.0, min(1.0, float(weighted_sum / total_weight)))

                safe_rationale = _sanitize_rationale_text(eval_data["rationale"])
                score_meta = {
                    "evaluator": "LLMJudgeEvaluator",
                    "judge_model_id": self.judge_model_id,
                    "anonymous_label": label,
                }
                if self.judge_provider_id:
                    score_meta["judge_provider_id"] = self.judge_provider_id

                score = EvaluationScore(
                    candidate_id=cid,
                    overall_score=overall_score,
                    dimension_scores=dim_scores,
                    rationale=safe_rationale,
                    metadata=score_meta,
                )
                scores_by_candidate_id[cid] = validate_evaluation_score(score)

        # Return scores deterministically in original input results order
        return [scores_by_candidate_id[r.candidate_id] for r in results]
