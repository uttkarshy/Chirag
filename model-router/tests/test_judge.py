"""Unit test suite for LLM Judge Evaluator (M3.11)."""

import json
import math
import pytest

from app.evaluation import (
    DEFAULT_JUDGE_RUBRIC,
    EvaluationDimension,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
    Evaluator,
    InvalidEvaluationDimensionError,
    InvalidJudgeResponseError,
    JudgeEvaluationError,
    JudgeExecutionError,
    JudgePromptBuilder,
    LLMJudgeEvaluator,
    SUPPORTED_JUDGE_DIMENSIONS,
    UnsupportedEvaluationDimensionError,
    get_anonymous_label,
)
from app.executor import ModelExecutor
from app.schema import ExecutionStatus, Modality, ModelExecutionRequest, ModelResult


class FakeJudgeExecutor(ModelExecutor):
    """Deterministic fake ModelExecutor simulating the judge model."""

    def __init__(
        self,
        output: str | None = None,
        should_fail: bool = False,
        failure_error: str = "Judge execution failure",
        exception_to_raise: Exception | None = None,
    ) -> None:
        self.output = output
        self.should_fail = should_fail
        self.failure_error = failure_error
        self.exception_to_raise = exception_to_raise
        self.captured_requests: list[ModelExecutionRequest] = []

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        self.captured_requests.append(request)

        if self.exception_to_raise is not None:
            raise self.exception_to_raise

        if self.should_fail:
            return ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=Modality.TEXT.value,
                error=self.failure_error,
            )

        return ModelResult(
            model_id=request.model_id,
            status=ExecutionStatus.SUCCESS.value,
            output=self.output,
            output_modality=Modality.TEXT.value,
            usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
            metadata={"mock_judge": "true"},
        )


# ============================================================================
# 1. Contract & Abstraction Tests
# ============================================================================


def test_judge_implements_evaluator():
    fake_exec = FakeJudgeExecutor(output="{}")
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="judge-model-1")
    assert isinstance(judge, Evaluator)


def test_judge_requires_injected_executor():
    with pytest.raises(TypeError) as exc_info:
        LLMJudgeEvaluator(executor=None, judge_model_id="judge-model-1")  # type: ignore
    assert "requires a ModelExecutor instance" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info2:
        fake_exec = FakeJudgeExecutor()
        LLMJudgeEvaluator(executor=fake_exec, judge_model_id="")
    assert "judge_model_id must be a non-empty string" in str(exc_info2.value)


# ============================================================================
# 2. Prompt Builder & Blind Evaluation Tests
# ============================================================================


def test_anonymous_labeling_deterministic():
    assert get_anonymous_label(0) == "Candidate A"
    assert get_anonymous_label(1) == "Candidate B"
    assert get_anonymous_label(25) == "Candidate Z"
    assert get_anonymous_label(26) == "Candidate AA"


def test_prompt_builder_blind_evaluation_no_id_leaks():
    builder = JudgePromptBuilder()
    task = EvaluationTask(
        task_id="t-blind",
        input="Explain the carbon cycle.",
        expected_output="Plants absorb CO2, respiration releases it.",
    )
    dimensions = [
        EvaluationDimension(name="correctness", weight=1.0),
        EvaluationDimension(name="relevance", weight=1.0),
    ]
    candidates = [
        {"label": "Candidate A", "output": "Plants take in carbon dioxide during photosynthesis."},
        {"label": "Candidate B", "output": "Carbon moves through ecosystems and atmosphere."},
    ]

    prompt = builder.build_prompt(task, candidates, dimensions)

    # Content must be present
    assert "Explain the carbon cycle." in prompt
    assert "Plants absorb CO2, respiration releases it." in prompt
    assert "correctness" in prompt
    assert "relevance" in prompt
    assert "Candidate A" in prompt
    assert "Candidate B" in prompt
    assert "Plants take in carbon dioxide" in prompt

    # Candidate identities must NOT appear
    assert "deepseek" not in prompt
    assert "openrouter" not in prompt
    assert "qwen" not in prompt
    assert "llama" not in prompt


def test_prompt_builder_deterministic_stability():
    builder = JudgePromptBuilder()
    task = EvaluationTask(task_id="t1", input="Input")
    dims = [EvaluationDimension(name="completeness")]
    cands = [{"label": "Candidate A", "output": "Output"}]

    prompt1 = builder.build_prompt(task, cands, dims)
    prompt2 = builder.build_prompt(task, cands, dims)
    assert prompt1 == prompt2


# ============================================================================
# 3. Judge Execution Request Properties Tests
# ============================================================================


def test_judge_model_execution_request_properties():
    judge_response = json.dumps({
        "evaluations": [
            {
                "candidate_id": "Candidate A",
                "scores": {"correctness": 0.95},
                "rationale": "Scientifically accurate.",
            }
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(
        executor=fake_exec,
        judge_model_id="meta-llama/llama-3.3-70b-instruct",
        judge_provider_id="openrouter",
        metadata={"judge_env": "production"},
    )

    task = EvaluationTask(task_id="t-req", input="Explain photosynthesis")
    res = EvaluationResult(
        candidate_id="cand-1",
        model_id="deepseek/deepseek-chat",
        provider_id="openrouter",
        status="success",
        output="Photosynthesis converts light into energy.",
    )
    dim = EvaluationDimension(name="correctness", weight=1.0)

    judge.evaluate(task, [res], [dim])

    assert len(fake_exec.captured_requests) == 1
    req = fake_exec.captured_requests[0]

    assert req.model_id == "meta-llama/llama-3.3-70b-instruct"
    assert req.metadata["provider_id"] == "openrouter"
    assert req.metadata["judge_env"] == "production"
    assert req.metadata["evaluation_role"] == "llm_judge"
    assert req.structured_output is True
    assert req.input_modalities == ["text"]
    assert req.output_modalities == ["text"]


# ============================================================================
# 4. Valid Responses & Semantic Dimensions Tests
# ============================================================================


def test_valid_response_single_candidate_all_four_dimensions():
    judge_response = json.dumps({
        "evaluations": [
            {
                "candidate_id": "Candidate A",
                "scores": {
                    "correctness": 0.90,
                    "relevance": 1.0,
                    "completeness": 0.80,
                    "instruction_following": 0.95,
                },
                "rationale": "High quality answer covering all aspects thoroughly.",
            }
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="judge-1")

    task = EvaluationTask(task_id="t-all-dims", input="Explain black holes.")
    res = EvaluationResult(
        candidate_id="cand-qwen",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="A black hole is a region of spacetime...",
    )
    dims = [
        EvaluationDimension(name="correctness", weight=1.0),
        EvaluationDimension(name="relevance", weight=1.0),
        EvaluationDimension(name="completeness", weight=1.0),
        EvaluationDimension(name="instruction_following", weight=1.0),
    ]

    scores = judge.evaluate(task, [res], dims)

    assert len(scores) == 1
    score = scores[0]
    assert score.candidate_id == "cand-qwen"
    assert score.dimension_scores["correctness"] == 0.90
    assert score.dimension_scores["relevance"] == 1.0
    assert score.dimension_scores["completeness"] == 0.80
    assert score.dimension_scores["instruction_following"] == 0.95
    # (0.90 + 1.0 + 0.80 + 0.95) / 4 = 3.65 / 4 = 0.9125
    assert score.overall_score == pytest.approx(0.9125, 0.0001)
    assert "High quality answer" in (score.rationale or "")


def test_valid_response_multiple_candidates_ordering_preserved():
    judge_response = json.dumps({
        "evaluations": [
            {
                "candidate_id": "Candidate A",
                "scores": {"correctness": 0.85},
                "rationale": "Good.",
            },
            {
                "candidate_id": "Candidate B",
                "scores": {"correctness": 0.95},
                "rationale": "Superb.",
            },
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="judge-1")

    task = EvaluationTask(task_id="t-multi", input="Prompt")
    res1 = EvaluationResult(candidate_id="cand-z", model_id="m1", provider_id="p1", status="success", output="Out 1")
    res2 = EvaluationResult(candidate_id="cand-a", model_id="m2", provider_id="p2", status="success", output="Out 2")
    dim = EvaluationDimension(name="correctness", weight=1.0)

    scores = judge.evaluate(task, [res1, res2], [dim])

    # Result order must match input results: [cand-z, cand-a]
    assert [s.candidate_id for s in scores] == ["cand-z", "cand-a"]
    assert scores[0].overall_score == 0.85
    assert scores[1].overall_score == 0.95


def test_valid_response_weighted_dimensions():
    judge_response = json.dumps({
        "evaluations": [
            {
                "candidate_id": "Candidate A",
                "scores": {
                    "correctness": 1.0,
                    "relevance": 0.5,
                },
                "rationale": "Accurate but partly off-topic.",
            }
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="judge-1")

    task = EvaluationTask(task_id="t-wt", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="Out")
    dims = [
        EvaluationDimension(name="correctness", weight=0.8),
        EvaluationDimension(name="relevance", weight=0.2),
    ]

    scores = judge.evaluate(task, [res], dims)
    # (1.0*0.8 + 0.5*0.2) / 1.0 = 0.9
    assert scores[0].overall_score == pytest.approx(0.9, 0.0001)


# ============================================================================
# 5. Invalid Responses & Strict Validation Tests
# ============================================================================


def test_invalid_response_malformed_json():
    fake_exec = FakeJudgeExecutor(output="This is not JSON at all.")
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t1", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="Out")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(InvalidJudgeResponseError) as exc_info:
        judge.evaluate(task, [res], [dim])
    assert "malformed JSON" in str(exc_info.value)


def test_invalid_response_missing_candidate():
    # Only Candidate A returned when A and B were expected
    judge_response = json.dumps({
        "evaluations": [
            {"candidate_id": "Candidate A", "scores": {"correctness": 0.9}, "rationale": "R"}
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t1", input="Prompt")
    res1 = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    res2 = EvaluationResult(candidate_id="c2", model_id="m2", provider_id="p2", status="success", output="2")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(InvalidJudgeResponseError) as exc_info:
        judge.evaluate(task, [res1, res2], [dim])
    assert "Missing evaluation for candidate 'Candidate B'" in str(exc_info.value)


def test_invalid_response_unknown_candidate():
    judge_response = json.dumps({
        "evaluations": [
            {"candidate_id": "Candidate Unknown", "scores": {"correctness": 0.9}, "rationale": "R"}
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t1", input="Prompt")
    res1 = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(InvalidJudgeResponseError) as exc_info:
        judge.evaluate(task, [res1], [dim])
    assert "Unknown candidate 'Candidate Unknown'" in str(exc_info.value)


def test_invalid_response_duplicate_candidate():
    judge_response = json.dumps({
        "evaluations": [
            {"candidate_id": "Candidate A", "scores": {"correctness": 0.9}, "rationale": "R1"},
            {"candidate_id": "Candidate A", "scores": {"correctness": 0.8}, "rationale": "R2"},
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t1", input="Prompt")
    res1 = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(InvalidJudgeResponseError) as exc_info:
        judge.evaluate(task, [res1], [dim])
    assert "Duplicate evaluation detected for candidate 'Candidate A'" in str(exc_info.value)


def test_invalid_response_missing_dimension():
    judge_response = json.dumps({
        "evaluations": [
            {"candidate_id": "Candidate A", "scores": {"correctness": 0.9}, "rationale": "R"}
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t1", input="Prompt")
    res1 = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dims = [
        EvaluationDimension(name="correctness"),
        EvaluationDimension(name="relevance"),  # Missing in judge response
    ]

    with pytest.raises(InvalidJudgeResponseError) as exc_info:
        judge.evaluate(task, [res1], dims)
    assert "Missing score for requested dimension 'relevance'" in str(exc_info.value)


def test_invalid_response_unknown_dimension():
    judge_response = json.dumps({
        "evaluations": [
            {"candidate_id": "Candidate A", "scores": {"correctness": 0.9, "creativity": 0.8}, "rationale": "R"}
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t1", input="Prompt")
    res1 = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dims = [EvaluationDimension(name="correctness")]

    with pytest.raises(InvalidJudgeResponseError) as exc_info:
        judge.evaluate(task, [res1], dims)
    assert "Unknown dimension 'creativity'" in str(exc_info.value)


def test_invalid_response_score_out_of_bounds():
    # Score > 1.0
    fake_exec1 = FakeJudgeExecutor(output=json.dumps({
        "evaluations": [{"candidate_id": "Candidate A", "scores": {"correctness": 1.5}, "rationale": "R"}]
    }))
    judge1 = LLMJudgeEvaluator(executor=fake_exec1, judge_model_id="j1")
    task = EvaluationTask(task_id="t1", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(InvalidJudgeResponseError) as exc1:
        judge1.evaluate(task, [res], [dim])
    assert "must be between 0.0 and 1.0" in str(exc1.value)

    # Score < 0.0
    fake_exec2 = FakeJudgeExecutor(output=json.dumps({
        "evaluations": [{"candidate_id": "Candidate A", "scores": {"correctness": -0.1}, "rationale": "R"}]
    }))
    judge2 = LLMJudgeEvaluator(executor=fake_exec2, judge_model_id="j1")
    with pytest.raises(InvalidJudgeResponseError) as exc2:
        judge2.evaluate(task, [res], [dim])
    assert "must be between 0.0 and 1.0" in str(exc2.value)


def test_invalid_response_nan_and_infinity():
    # NaN
    fake_exec1 = FakeJudgeExecutor(output='{"evaluations": [{"candidate_id": "Candidate A", "scores": {"correctness": NaN}, "rationale": "R"}]}')
    judge1 = LLMJudgeEvaluator(executor=fake_exec1, judge_model_id="j1")
    task = EvaluationTask(task_id="t1", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(InvalidJudgeResponseError):
        judge1.evaluate(task, [res], [dim])


def test_invalid_response_missing_rationale():
    fake_exec = FakeJudgeExecutor(output=json.dumps({
        "evaluations": [{"candidate_id": "Candidate A", "scores": {"correctness": 0.9}, "rationale": ""}]
    }))
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")
    task = EvaluationTask(task_id="t1", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(InvalidJudgeResponseError) as exc_info:
        judge.evaluate(task, [res], [dim])
    assert "must contain a non-empty rationale string" in str(exc_info.value)


def test_empty_candidates_and_empty_dimensions():
    fake_exec = FakeJudgeExecutor()
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")
    task = EvaluationTask(task_id="t1", input="Prompt")
    dim = EvaluationDimension(name="correctness")

    # Empty results -> returns []
    assert judge.evaluate(task, [], [dim]) == []

    # Empty dimensions -> raises InvalidEvaluationDimensionError
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    with pytest.raises(InvalidEvaluationDimensionError):
        judge.evaluate(task, [res], [])


def test_unsupported_dimension_rejected():
    fake_exec = FakeJudgeExecutor()
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")
    task = EvaluationTask(task_id="t1", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dim = EvaluationDimension(name="execution_speed")  # Not supported

    with pytest.raises(UnsupportedEvaluationDimensionError) as exc_info:
        judge.evaluate(task, [res], [dim])
    assert "is not supported by LLMJudgeEvaluator" in str(exc_info.value)


def test_failed_judge_execution_and_empty_output():
    # Failed execution
    fake_exec_failed = FakeJudgeExecutor(should_fail=True, failure_error="CUDA out of memory on judge")
    judge_failed = LLMJudgeEvaluator(executor=fake_exec_failed, judge_model_id="j1")
    task = EvaluationTask(task_id="t1", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="1")
    dim = EvaluationDimension(name="correctness")

    with pytest.raises(JudgeExecutionError) as exc1:
        judge_failed.evaluate(task, [res], [dim])
    assert "Judge model execution failed" in str(exc1.value)

    # Empty output
    fake_exec_empty = FakeJudgeExecutor(output="")
    judge_empty = LLMJudgeEvaluator(executor=fake_exec_empty, judge_model_id="j1")
    with pytest.raises(InvalidJudgeResponseError) as exc2:
        judge_empty.evaluate(task, [res], [dim])
    assert "Judge model returned empty output" in str(exc2.value)


# ============================================================================
# 6. Failure Isolation Among Candidates Tests
# ============================================================================


def test_failure_isolation_failed_candidate_bypasses_judge():
    """Verify failed candidate B receives 0.0 directly while candidates A and C are judged."""
    judge_response = json.dumps({
        "evaluations": [
            {"candidate_id": "Candidate A", "scores": {"correctness": 0.8}, "rationale": "Good."},
            {"candidate_id": "Candidate B", "scores": {"correctness": 0.95}, "rationale": "Excellent."},
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t1", input="Prompt")
    res_a = EvaluationResult(candidate_id="cand-a", model_id="m1", provider_id="p1", status="success", output="Out A")
    res_b = EvaluationResult(candidate_id="cand-b", model_id="m2", provider_id="p1", status="failed", error="Rate limited")
    res_c = EvaluationResult(candidate_id="cand-c", model_id="m3", provider_id="p1", status="success", output="Out C")
    dim = EvaluationDimension(name="correctness", weight=1.0)

    scores = judge.evaluate(task, [res_a, res_b, res_c], [dim])

    assert len(scores) == 3
    # Order preserved: [cand-a, cand-b, cand-c]
    assert scores[0].candidate_id == "cand-a"
    assert scores[0].overall_score == 0.8

    assert scores[1].candidate_id == "cand-b"
    assert scores[1].overall_score == 0.0
    assert scores[1].dimension_scores["correctness"] == 0.0
    assert "execution failed" in (scores[1].rationale or "")

    assert scores[2].candidate_id == "cand-c"
    assert scores[2].overall_score == 0.95

    # Only A and C should have been passed to the judge prompt
    prompt = fake_exec.captured_requests[0].input
    assert "Out A" in prompt
    assert "Out C" in prompt
    assert "Rate limited" not in prompt


# ============================================================================
# 7. Security & Credential Redaction Tests
# ============================================================================


def test_security_rationale_redacts_credentials():
    judge_response = json.dumps({
        "evaluations": [
            {
                "candidate_id": "Candidate A",
                "scores": {"correctness": 0.8},
                "rationale": "Good answer but leaks token sk-or-v1-abcdef123456789 and Bearer secrettokenxyz.",
            }
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="j1")

    task = EvaluationTask(task_id="t-sec", input="Prompt")
    res = EvaluationResult(candidate_id="c1", model_id="m1", provider_id="p1", status="success", output="Out")
    dim = EvaluationDimension(name="correctness")

    scores = judge.evaluate(task, [res], [dim])
    rationale = scores[0].rationale or ""

    assert "abcdef123456789" not in rationale
    assert "secrettokenxyz" not in rationale
    assert "[REDACTED]" in rationale


# ============================================================================
# 8. Evaluation Summary & Ranking Integration Tests
# ============================================================================


def test_evaluate_and_summarize_populates_ranked_summary():
    judge_response = json.dumps({
        "evaluations": [
            {"candidate_id": "Candidate A", "scores": {"correctness": 0.70}, "rationale": "Fair."},
            {"candidate_id": "Candidate B", "scores": {"correctness": 0.95}, "rationale": "Superior."},
        ]
    })
    fake_exec = FakeJudgeExecutor(output=judge_response)
    judge = LLMJudgeEvaluator(executor=fake_exec, judge_model_id="judge-1")

    task = EvaluationTask(task_id="t-sum", input="Explain relativity")
    res1 = EvaluationResult(candidate_id="cand-1", model_id="m1", provider_id="p1", status="success", output="Relativity 1")
    res2 = EvaluationResult(candidate_id="cand-2", model_id="m2", provider_id="p2", status="success", output="Relativity 2")
    dim = EvaluationDimension(name="correctness", weight=1.0)

    summary = judge.evaluate_and_summarize(
        task=task,
        results=[res1, res2],
        dimensions=[dim],
        metadata={"domain": "physics"},
    )

    assert isinstance(summary, EvaluationSummary)
    assert summary.task_id == "t-sum"
    assert len(summary.results) == 2
    assert len(summary.scores) == 2

    # Winner must be cand-2 (0.95 vs 0.70)
    assert summary.winner_candidate_id == "cand-2"
    assert summary.scores[0].candidate_id == "cand-2"
    assert summary.scores[0].overall_score == 0.95
    assert summary.metadata["domain"] == "physics"
