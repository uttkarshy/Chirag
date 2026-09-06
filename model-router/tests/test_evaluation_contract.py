import random
import pytest
from pydantic import ValidationError

from app.evaluation import (
    EvaluationCandidate,
    EvaluationDimension,
    EvaluationError,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
    InvalidEvaluationCandidateError,
    InvalidEvaluationDimensionError,
    InvalidEvaluationResultError,
    InvalidEvaluationScoreError,
    InvalidEvaluationSummaryError,
    InvalidEvaluationTaskError,
    build_evaluation_summary,
    evaluation_result_from_model_result,
    rank_scores,
    select_winner,
    validate_evaluation_candidate,
    validate_evaluation_dimension,
    validate_evaluation_result,
    validate_evaluation_score,
    validate_evaluation_summary,
    validate_evaluation_task,
)
from app.schema import ExecutionStatus, Modality, ModelResult


# ============================================================================
# 1. EvaluationTask Tests
# ============================================================================


def test_evaluation_task_valid():
    task = EvaluationTask(
        task_id="eval-task-001",
        input="Summarize this financial quarterly report.",
        expected_output="A concise three-bullet summary.",
        input_modalities=["text"],
        output_modalities=["text"],
        structured_output=True,
        metadata={"category": "finance", "difficulty": "medium"},
    )
    assert task.task_id == "eval-task-001"
    assert task.input == "Summarize this financial quarterly report."
    assert task.expected_output == "A concise three-bullet summary."
    assert task.input_modalities == ["text"]
    assert task.output_modalities == ["text"]
    assert task.structured_output is True
    assert task.metadata["category"] == "finance"
    validate_evaluation_task(task)


def test_evaluation_task_empty_identifiers_rejected():
    with pytest.raises(ValidationError):
        EvaluationTask(task_id="", input="Some input")

    with pytest.raises(ValidationError):
        EvaluationTask(task_id="task-1", input="")


def test_evaluation_task_invalid_modalities():
    with pytest.raises(ValidationError):
        EvaluationTask(task_id="task-1", input="Input", input_modalities=["hologram"])

    with pytest.raises(ValidationError):
        EvaluationTask(task_id="task-1", input="Input", input_modalities=[])


def test_evaluation_task_forbids_extra_fields():
    with pytest.raises(ValidationError):
        EvaluationTask(task_id="task-1", input="Input", unknown_field="unexpected")


# ============================================================================
# 2. EvaluationCandidate Tests
# ============================================================================


def test_evaluation_candidate_valid():
    cand = EvaluationCandidate(
        candidate_id="cand-openrouter-deepseek",
        model_id="deepseek/deepseek-chat",
        provider_id="openrouter",
        metadata={"temperature": "0.2"},
    )
    assert cand.candidate_id == "cand-openrouter-deepseek"
    assert cand.model_id == "deepseek/deepseek-chat"
    assert cand.provider_id == "openrouter"
    assert cand.metadata["temperature"] == "0.2"
    validate_evaluation_candidate(cand)


def test_evaluation_candidate_same_model_multiple_providers():
    cand1 = EvaluationCandidate(
        candidate_id="cand-1",
        model_id="deepseek/deepseek-chat",
        provider_id="openrouter",
    )
    cand2 = EvaluationCandidate(
        candidate_id="cand-2",
        model_id="deepseek/deepseek-chat",
        provider_id="nim",
    )
    assert cand1.model_id == cand2.model_id
    assert cand1.provider_id != cand2.provider_id
    assert cand1.candidate_id != cand2.candidate_id


def test_evaluation_candidate_empty_fields_rejected():
    with pytest.raises(ValidationError):
        EvaluationCandidate(candidate_id="", model_id="m1", provider_id="p1")

    with pytest.raises(ValidationError):
        EvaluationCandidate(candidate_id="c1", model_id="", provider_id="p1")

    with pytest.raises(ValidationError):
        EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="")


def test_evaluation_candidate_forbids_extra_fields():
    with pytest.raises(ValidationError):
        EvaluationCandidate(
            candidate_id="c1",
            model_id="m1",
            provider_id="p1",
            extra_key="not_allowed",
        )


# ============================================================================
# 3. EvaluationResult Tests
# ============================================================================


def test_evaluation_result_success():
    res = EvaluationResult(
        candidate_id="cand-1",
        model_id="meta-llama/llama-3.3-70b-instruct",
        provider_id="openrouter",
        status=EvaluationStatus.SUCCESS.value,
        output="The answer is 42.",
        latency_ms=124.5,
        usage={"total_tokens": 150},
        metadata={"finish_reason": "stop"},
    )
    assert res.status == "success"
    assert res.output == "The answer is 42."
    assert res.error is None
    assert res.latency_ms == 124.5
    validate_evaluation_result(res)


def test_evaluation_result_failed():
    res = EvaluationResult(
        candidate_id="cand-1",
        model_id="meta-llama/llama-3.3-70b-instruct",
        provider_id="openrouter",
        status=EvaluationStatus.FAILED.value,
        error="Rate limit exceeded",
        latency_ms=45.0,
    )
    assert res.status == "failed"
    assert res.error == "Rate limit exceeded"
    assert res.output is None
    validate_evaluation_result(res)


def test_evaluation_result_skipped():
    res = EvaluationResult(
        candidate_id="cand-1",
        model_id="meta-llama/llama-3.3-70b-instruct",
        provider_id="openrouter",
        status=EvaluationStatus.SKIPPED.value,
        metadata={"reason": "precondition not met"},
    )
    assert res.status == "skipped"
    validate_evaluation_result(res)


def test_evaluation_result_invalid_status():
    with pytest.raises(ValidationError):
        EvaluationResult(
            candidate_id="c1",
            model_id="m1",
            provider_id="p1",
            status="pending",
        )


def test_evaluation_result_negative_latency_rejected():
    with pytest.raises(ValidationError):
        EvaluationResult(
            candidate_id="c1",
            model_id="m1",
            provider_id="p1",
            status="success",
            output="ok",
            latency_ms=-1.0,
        )


def test_evaluation_result_validation_invariants():
    # Success requires output
    res_no_output = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output=None,
    )
    with pytest.raises(InvalidEvaluationResultError):
        validate_evaluation_result(res_no_output)

    # Failed requires error
    res_no_error = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="failed",
        error=None,
    )
    with pytest.raises(InvalidEvaluationResultError):
        validate_evaluation_result(res_no_error)


def test_evaluation_result_forbids_extra_fields():
    with pytest.raises(ValidationError):
        EvaluationResult(
            candidate_id="c1",
            model_id="m1",
            provider_id="p1",
            status="success",
            output="ok",
            bogus="forbidden",
        )


# ============================================================================
# 4. EvaluationDimension Tests
# ============================================================================


def test_evaluation_dimension_valid():
    dim = EvaluationDimension(
        name="correctness",
        description="Factual and logical accuracy of generated output.",
        weight=2.0,
    )
    assert dim.name == "correctness"
    assert dim.weight == 2.0
    validate_evaluation_dimension(dim)


def test_evaluation_dimension_negative_weight_rejected():
    with pytest.raises(ValidationError):
        EvaluationDimension(name="speed", weight=-0.5)


def test_evaluation_dimension_empty_name_rejected():
    with pytest.raises(ValidationError):
        EvaluationDimension(name="", weight=1.0)


def test_evaluation_dimension_forbids_extra_fields():
    with pytest.raises(ValidationError):
        EvaluationDimension(name="dim", weight=1.0, extra="none")


# ============================================================================
# 5. EvaluationScore Tests
# ============================================================================


def test_evaluation_score_valid():
    score = EvaluationScore(
        candidate_id="cand-1",
        overall_score=0.92,
        dimension_scores={"correctness": 0.95, "brevity": 0.85},
        rationale="Clear and factually sound response.",
    )
    assert score.candidate_id == "cand-1"
    assert score.overall_score == 0.92
    assert score.dimension_scores["correctness"] == 0.95
    validate_evaluation_score(score)


def test_evaluation_score_range_validation():
    # overall_score must be in [0.0, 1.0]
    with pytest.raises(ValidationError):
        EvaluationScore(candidate_id="cand-1", overall_score=1.05)

    with pytest.raises(ValidationError):
        EvaluationScore(candidate_id="cand-1", overall_score=-0.1)

    # dimension_scores values must be in [0.0, 1.0]
    with pytest.raises(ValidationError):
        EvaluationScore(
            candidate_id="cand-1",
            overall_score=0.5,
            dimension_scores={"accuracy": 1.2},
        )


def test_evaluation_score_forbids_extra_fields():
    with pytest.raises(ValidationError):
        EvaluationScore(
            candidate_id="cand-1",
            overall_score=0.8,
            extra_field="unsupported",
        )


# ============================================================================
# 6. EvaluationSummary Tests
# ============================================================================


def test_evaluation_summary_valid():
    task = EvaluationTask(task_id="task-1", input="Explain gravity.")
    res1 = EvaluationResult(
        candidate_id="cand-1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="Gravity attracts masses.",
    )
    res2 = EvaluationResult(
        candidate_id="cand-2",
        model_id="m2",
        provider_id="p2",
        status="success",
        output="Gravity curves spacetime.",
    )
    sc1 = EvaluationScore(candidate_id="cand-1", overall_score=0.80)
    sc2 = EvaluationScore(candidate_id="cand-2", overall_score=0.95)

    summary = EvaluationSummary(
        task_id="task-1",
        results=[res1, res2],
        scores=[sc2, sc1],
        winner_candidate_id="cand-2",
        metadata={"benchmark": "physics-v1"},
    )
    assert summary.task_id == "task-1"
    assert summary.winner_candidate_id == "cand-2"
    assert len(summary.results) == 2
    assert len(summary.scores) == 2
    validate_evaluation_summary(summary)


def test_evaluation_summary_rejects_failed_candidate_as_winner():
    res1 = EvaluationResult(
        candidate_id="cand-failed",
        model_id="m1",
        provider_id="p1",
        status="failed",
        error="Network error",
    )
    summary = EvaluationSummary(
        task_id="task-1",
        results=[res1],
        scores=[EvaluationScore(candidate_id="cand-failed", overall_score=0.5)],
        winner_candidate_id="cand-failed",
    )
    with pytest.raises(InvalidEvaluationSummaryError):
        validate_evaluation_summary(summary)


def test_evaluation_summary_forbids_extra_fields():
    with pytest.raises(ValidationError):
        EvaluationSummary(task_id="task-1", extra="not_allowed")


# ============================================================================
# 7. Ranking & Determinism Tests
# ============================================================================


def test_rank_scores_descending_by_overall_score():
    s1 = EvaluationScore(candidate_id="cand-1", overall_score=0.70)
    s2 = EvaluationScore(candidate_id="cand-2", overall_score=0.95)
    s3 = EvaluationScore(candidate_id="cand-3", overall_score=0.85)

    ranked = rank_scores([s1, s2, s3])
    assert [s.candidate_id for s in ranked] == ["cand-2", "cand-3", "cand-1"]
    assert [s.overall_score for s in ranked] == [0.95, 0.85, 0.70]


def test_rank_scores_deterministic_tie_breaking():
    # Same score: tie-breaker is candidate_id ascending (alphabetical)
    s_zeta = EvaluationScore(candidate_id="zeta-candidate", overall_score=0.90)
    s_alpha = EvaluationScore(candidate_id="alpha-candidate", overall_score=0.90)
    s_beta = EvaluationScore(candidate_id="beta-candidate", overall_score=0.90)

    # Regardless of input ordering, alphabetical tie-breaking must be exact
    for _ in range(10):
        shuffled = [s_zeta, s_alpha, s_beta]
        random.shuffle(shuffled)
        ranked = rank_scores(shuffled)
        assert [s.candidate_id for s in ranked] == [
            "alpha-candidate",
            "beta-candidate",
            "zeta-candidate",
        ]


def test_rank_scores_mixed_scores_and_ties():
    s_a = EvaluationScore(candidate_id="cand-a", overall_score=0.85)
    s_b = EvaluationScore(candidate_id="cand-b", overall_score=0.85)
    s_c = EvaluationScore(candidate_id="cand-c", overall_score=0.95)
    s_d = EvaluationScore(candidate_id="cand-d", overall_score=0.60)

    ranked = rank_scores([s_a, s_b, s_c, s_d])
    assert [s.candidate_id for s in ranked] == ["cand-c", "cand-a", "cand-b", "cand-d"]


# ============================================================================
# 8. Winner Selection Tests
# ============================================================================


def test_select_winner_basic():
    s1 = EvaluationScore(candidate_id="cand-1", overall_score=0.80)
    s2 = EvaluationScore(candidate_id="cand-2", overall_score=0.92)

    winner = select_winner([s1, s2])
    assert winner == "cand-2"


def test_select_winner_tie_break():
    s1 = EvaluationScore(candidate_id="cand-z", overall_score=0.90)
    s2 = EvaluationScore(candidate_id="cand-a", overall_score=0.90)

    assert select_winner([s1, s2]) == "cand-a"


def test_select_winner_excludes_failed_candidate():
    r1 = EvaluationResult(
        candidate_id="cand-1",
        model_id="m1",
        provider_id="p1",
        status="failed",
        error="Crash",
    )
    r2 = EvaluationResult(
        candidate_id="cand-2",
        model_id="m2",
        provider_id="p2",
        status="success",
        output="Result",
    )
    s1 = EvaluationScore(candidate_id="cand-1", overall_score=0.99)
    s2 = EvaluationScore(candidate_id="cand-2", overall_score=0.75)

    # Even though cand-1 scored higher, it failed execution and must not win
    winner = select_winner([s1, s2], results=[r1, r2])
    assert winner == "cand-2"


def test_select_winner_no_winner_when_all_fail():
    r1 = EvaluationResult(
        candidate_id="cand-1",
        model_id="m1",
        provider_id="p1",
        status="failed",
        error="Timeout",
    )
    r2 = EvaluationResult(
        candidate_id="cand-2",
        model_id="m2",
        provider_id="p2",
        status="failed",
        error="OOM",
    )
    s1 = EvaluationScore(candidate_id="cand-1", overall_score=0.50)
    s2 = EvaluationScore(candidate_id="cand-2", overall_score=0.50)

    winner = select_winner([s1, s2], results=[r1, r2])
    assert winner is None


def test_select_winner_empty_scores_returns_none():
    assert select_winner([]) is None


# ============================================================================
# 9. Summary Builder Tests
# ============================================================================


def test_build_evaluation_summary():
    task = EvaluationTask(task_id="task-calc", input="Calculate 2 + 2")
    c1 = EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1")
    c2 = EvaluationCandidate(candidate_id="c2", model_id="m2", provider_id="p2")

    r1 = EvaluationResult(
        candidate_id=c1.candidate_id,
        model_id=c1.model_id,
        provider_id=c1.provider_id,
        status="success",
        output="4",
        latency_ms=10.0,
    )
    r2 = EvaluationResult(
        candidate_id=c2.candidate_id,
        model_id=c2.model_id,
        provider_id=c2.provider_id,
        status="success",
        output="Four",
        latency_ms=15.0,
    )

    s1 = EvaluationScore(candidate_id="c1", overall_score=1.0)
    s2 = EvaluationScore(candidate_id="c2", overall_score=0.8)

    summary = build_evaluation_summary(
        task=task,
        results=[r1, r2],
        scores=[s2, s1],
        metadata={"run_type": "unit-test"},
    )

    assert summary.task_id == "task-calc"
    assert summary.winner_candidate_id == "c1"
    assert [s.candidate_id for s in summary.scores] == ["c1", "c2"]
    assert summary.metadata["run_type"] == "unit-test"


# ============================================================================
# 10. Bridge to ModelResult Tests
# ============================================================================


def test_evaluation_result_from_model_result_success():
    cand = EvaluationCandidate(
        candidate_id="cand-qwen",
        model_id="qwen/qwen-2.5-coder-32b-instruct",
        provider_id="openrouter",
    )
    model_res = ModelResult(
        model_id="qwen/qwen-2.5-coder-32b-instruct",
        status=ExecutionStatus.SUCCESS.value,
        output="def add(a, b): return a + b",
        output_modality=Modality.TEXT.value,
        usage={"prompt_tokens": 10, "completion_tokens": 15},
        metadata={"cached": "true"},
    )

    eval_res = evaluation_result_from_model_result(
        candidate=cand,
        model_result=model_res,
        latency_ms=85.2,
        metadata={"evaluator": "test_suite"},
    )

    assert eval_res.candidate_id == "cand-qwen"
    assert eval_res.model_id == "qwen/qwen-2.5-coder-32b-instruct"
    assert eval_res.provider_id == "openrouter"
    assert eval_res.status == EvaluationStatus.SUCCESS.value
    assert eval_res.output == "def add(a, b): return a + b"
    assert eval_res.error is None
    assert eval_res.latency_ms == 85.2
    assert eval_res.usage["prompt_tokens"] == 10
    assert eval_res.metadata["cached"] == "true"
    assert eval_res.metadata["evaluator"] == "test_suite"


def test_evaluation_result_from_model_result_failed():
    cand = EvaluationCandidate(
        candidate_id="cand-qwen",
        model_id="qwen/qwen-2.5-coder-32b-instruct",
        provider_id="openrouter",
    )
    model_res = ModelResult(
        model_id="qwen/qwen-2.5-coder-32b-instruct",
        status=ExecutionStatus.FAILED.value,
        output=None,
        output_modality=Modality.TEXT.value,
        error="HTTP 500: Server Error",
    )

    eval_res = evaluation_result_from_model_result(
        candidate=cand,
        model_result=model_res,
        latency_ms=12.0,
    )

    assert eval_res.status == EvaluationStatus.FAILED.value
    assert eval_res.error == "HTTP 500: Server Error"
    assert eval_res.output is None


def test_evaluation_result_from_model_result_model_mismatch_raises():
    cand = EvaluationCandidate(
        candidate_id="cand-1",
        model_id="model-expected",
        provider_id="openrouter",
    )
    model_res = ModelResult(
        model_id="model-unexpected",
        status=ExecutionStatus.SUCCESS.value,
        output="output",
        output_modality=Modality.TEXT.value,
    )

    with pytest.raises(InvalidEvaluationResultError) as exc_info:
        evaluation_result_from_model_result(candidate=cand, model_result=model_res)
    assert "does not match ModelResult model_id" in str(exc_info.value)


# ============================================================================
# 11. Architecture Invariant Tests
# ============================================================================


def test_model_result_unmodified_by_evaluation_contracts():
    """Verify ModelResult remains independent of evaluation schemas."""
    # ModelResult should NOT have evaluation fields
    assert not hasattr(ModelResult, "candidate_id")
    assert not hasattr(ModelResult, "overall_score")
    assert not hasattr(ModelResult, "winner_candidate_id")


def test_evaluation_layer_requires_zero_network_or_api_keys(monkeypatch):
    """Verify entire evaluation module functions offline without API keys."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    task = EvaluationTask(task_id="t-offline", input="test")
    c1 = EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1")
    res = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="ok",
    )
    score = EvaluationScore(candidate_id="c1", overall_score=1.0)
    summary = build_evaluation_summary(task=task, results=[res], scores=[score])

    assert summary.winner_candidate_id == "c1"
