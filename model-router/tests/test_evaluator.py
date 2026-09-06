"""Tests for Intelligent Evaluation Framework (M3.10)."""

import pytest

from app.evaluation import (
    DeterministicEvaluator,
    EvaluationDimension,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
    Evaluator,
    InvalidEvaluationDimensionError,
    UnsupportedEvaluationDimensionError,
)


# ============================================================================
# 1. Evaluator Abstraction Tests
# ============================================================================


def test_evaluator_is_abstract():
    """Verify Evaluator ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Evaluator()  # type: ignore


def test_deterministic_evaluator_inherits_and_implements_interface():
    evaluator = DeterministicEvaluator()
    assert isinstance(evaluator, Evaluator)


def test_evaluator_does_not_execute_models():
    """Evaluator only consumes pre-existing EvaluationResult objects without running inference."""
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t1", input="What is 2+2?", expected_output="4")
    res = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="4",
    )
    dim = EvaluationDimension(name="execution_success", weight=1.0)

    # Evaluates without any executor or network calls
    scores = evaluator.evaluate(task, [res], [dim])
    assert len(scores) == 1
    assert scores[0].candidate_id == "c1"
    assert scores[0].overall_score == 1.0


# ============================================================================
# 2. Deterministic Dimensions Tests
# ============================================================================


def test_dimension_execution_success():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-exec", input="Task")
    dim = EvaluationDimension(name="execution_success", weight=1.0)

    res_success = EvaluationResult(
        candidate_id="c-ok",
        model_id="m1",
        provider_id="p1",
        status=EvaluationStatus.SUCCESS.value,
        output="Done",
    )
    res_failed = EvaluationResult(
        candidate_id="c-fail",
        model_id="m2",
        provider_id="p1",
        status=EvaluationStatus.FAILED.value,
        error="Network failure",
    )

    scores = evaluator.evaluate(task, [res_success, res_failed], [dim])
    assert len(scores) == 2
    assert scores[0].candidate_id == "c-ok"
    assert scores[0].dimension_scores["execution_success"] == 1.0
    assert scores[0].overall_score == 1.0

    assert scores[1].candidate_id == "c-fail"
    assert scores[1].dimension_scores["execution_success"] == 0.0
    assert scores[1].overall_score == 0.0


def test_dimension_output_present():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-out", input="Task")
    dim = EvaluationDimension(name="output_present", weight=1.0)

    res_present = EvaluationResult(
        candidate_id="c-pres",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="Non-empty response",
    )
    res_empty = EvaluationResult(
        candidate_id="c-empty",
        model_id="m2",
        provider_id="p1",
        status="failed",
        output=None,
        error="No output",
    )

    scores = evaluator.evaluate(task, [res_present, res_empty], [dim])
    assert scores[0].dimension_scores["output_present"] == 1.0
    assert scores[1].dimension_scores["output_present"] == 0.0


def test_dimension_exact_match_success_and_mismatch():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(
        task_id="t-exact",
        input="What is the answer to everything?",
        expected_output="42",
    )
    dim = EvaluationDimension(name="exact_match", weight=1.0)

    res_match = EvaluationResult(
        candidate_id="c-match",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="  42 \n",  # Stripped match
    )
    res_mismatch = EvaluationResult(
        candidate_id="c-mismatch",
        model_id="m2",
        provider_id="p1",
        status="success",
        output="41",
    )

    scores = evaluator.evaluate(task, [res_match, res_mismatch], [dim])
    assert scores[0].dimension_scores["exact_match"] == 1.0
    assert scores[0].overall_score == 1.0

    assert scores[1].dimension_scores["exact_match"] == 0.0
    assert scores[1].overall_score == 0.0


def test_dimension_exact_match_rejected_when_expected_output_absent():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-no-exp", input="Tell a creative joke", expected_output=None)
    dim = EvaluationDimension(name="exact_match", weight=1.0)

    res = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="A funny joke",
    )

    with pytest.raises(UnsupportedEvaluationDimensionError) as exc_info:
        evaluator.evaluate(task, [res], [dim])
    assert "task.expected_output is None" in str(exc_info.value)


def test_unsupported_dimension_rejected():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-unsupported", input="Task")
    dim_semantic = EvaluationDimension(name="semantic_similarity", weight=1.0)

    res = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="Result",
    )

    with pytest.raises(UnsupportedEvaluationDimensionError) as exc_info:
        evaluator.evaluate(task, [res], [dim_semantic])
    assert "is not supported by DeterministicEvaluator" in str(exc_info.value)


# ============================================================================
# 3. Score Aggregation & Weighting Tests
# ============================================================================


def test_weighted_dimension_aggregation():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-weighted", input="Task", expected_output="42")

    # Dimensions with unequal weights
    # execution_success: weight 0.2
    # output_present: weight 0.3
    # exact_match: weight 0.5
    dims = [
        EvaluationDimension(name="execution_success", weight=0.2),
        EvaluationDimension(name="output_present", weight=0.3),
        EvaluationDimension(name="exact_match", weight=0.5),
    ]

    # Candidate matches execution and presence, but fails exact match
    # Expected overall: (1.0*0.2 + 1.0*0.3 + 0.0*0.5) / (0.2 + 0.3 + 0.5) = 0.5 / 1.0 = 0.5
    res = EvaluationResult(
        candidate_id="c-partial",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="100",
    )

    scores = evaluator.evaluate(task, [res], dims)
    assert scores[0].overall_score == pytest.approx(0.5, 0.001)
    assert scores[0].dimension_scores["execution_success"] == 1.0
    assert scores[0].dimension_scores["output_present"] == 1.0
    assert scores[0].dimension_scores["exact_match"] == 0.0


def test_zero_weight_dimensions():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-zero-wt", input="Task")
    dim = EvaluationDimension(name="execution_success", weight=0.0)

    res = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="Output",
    )

    scores = evaluator.evaluate(task, [res], [dim])
    assert scores[0].overall_score == 0.0
    assert scores[0].dimension_scores["execution_success"] == 1.0


def test_empty_dimensions_raises_error():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-empty-dim", input="Task")
    res = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="Output",
    )

    with pytest.raises(InvalidEvaluationDimensionError):
        evaluator.evaluate(task, [res], [])


def test_empty_results_returns_empty_scores():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-empty-res", input="Task")
    dim = EvaluationDimension(name="execution_success", weight=1.0)

    scores = evaluator.evaluate(task, [], [dim])
    assert scores == []


# ============================================================================
# 4. Multiple Candidates & Failure Isolation Tests
# ============================================================================


def test_multiple_candidates_with_failure_isolation():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-fail-iso", input="Calculate 5 * 5", expected_output="25")
    dims = [
        EvaluationDimension(name="execution_success", weight=0.5),
        EvaluationDimension(name="exact_match", weight=0.5),
    ]

    res_a = EvaluationResult(
        candidate_id="cand-a",
        model_id="m1",
        provider_id="openrouter",
        status="success",
        output="25",
    )
    res_b = EvaluationResult(
        candidate_id="cand-b",
        model_id="m2",
        provider_id="openrouter",
        status="failed",
        error="CUDA out of memory",
    )
    res_c = EvaluationResult(
        candidate_id="cand-c",
        model_id="m3",
        provider_id="openrouter",
        status="success",
        output="26",
    )

    scores = evaluator.evaluate(task, [res_a, res_b, res_c], dims)

    assert len(scores) == 3
    # Candidate A: 1.0 (success) + 1.0 (exact match) -> 1.0
    assert scores[0].candidate_id == "cand-a"
    assert scores[0].overall_score == 1.0

    # Candidate B: 0.0 (failed) + 0.0 (exact match) -> 0.0
    assert scores[1].candidate_id == "cand-b"
    assert scores[1].overall_score == 0.0
    assert "failed" in (scores[1].rationale or "")

    # Candidate C: 1.0 (success) + 0.0 (exact match) -> 0.5
    assert scores[2].candidate_id == "cand-c"
    assert scores[2].overall_score == 0.5


# ============================================================================
# 5. Summary Generation & Deterministic Ranking Tests
# ============================================================================


def test_evaluate_and_summarize_populates_scores_and_winner():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-sum", input="Solve x + 3 = 10", expected_output="7")
    dim = EvaluationDimension(name="exact_match", weight=1.0)

    res1 = EvaluationResult(
        candidate_id="cand-1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="8",
    )
    res2 = EvaluationResult(
        candidate_id="cand-2",
        model_id="m2",
        provider_id="p2",
        status="success",
        output="7",
    )

    summary = evaluator.evaluate_and_summarize(
        task=task,
        results=[res1, res2],
        dimensions=[dim],
        metadata={"suite": "algebra"},
    )

    assert isinstance(summary, EvaluationSummary)
    assert summary.task_id == "t-sum"
    assert len(summary.results) == 2
    assert len(summary.scores) == 2

    # Winner must be cand-2 (exact match 1.0 vs 0.0)
    assert summary.winner_candidate_id == "cand-2"
    assert summary.scores[0].candidate_id == "cand-2"
    assert summary.scores[0].overall_score == 1.0
    assert summary.metadata["suite"] == "algebra"


def test_evaluate_and_summarize_deterministic_tie_breaking():
    """Exact ties broken alphabetically by candidate_id."""
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-tie", input="Output yes", expected_output="yes")
    dim = EvaluationDimension(name="exact_match", weight=1.0)

    res_z = EvaluationResult(
        candidate_id="zeta-cand",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="yes",
    )
    res_a = EvaluationResult(
        candidate_id="alpha-cand",
        model_id="m2",
        provider_id="p2",
        status="success",
        output="yes",
    )

    summary = evaluator.evaluate_and_summarize(task, [res_z, res_a], [dim])
    assert summary.winner_candidate_id == "alpha-cand"
    assert [s.candidate_id for s in summary.scores] == ["alpha-cand", "zeta-cand"]


def test_evaluate_and_summarize_preserves_original_results():
    """Verify input EvaluationResult objects are not mutated."""
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-imm", input="Task")
    dim = EvaluationDimension(name="execution_success", weight=1.0)

    res = EvaluationResult(
        candidate_id="c1",
        model_id="m1",
        provider_id="p1",
        status="success",
        output="Immutable",
        latency_ms=12.5,
        usage={"tokens": 10},
    )

    summary = evaluator.evaluate_and_summarize(task, [res], [dim])

    # Result in summary is identical
    assert summary.results[0].candidate_id == res.candidate_id
    assert summary.results[0].output == res.output
    assert summary.results[0].latency_ms == res.latency_ms
    assert summary.results[0].usage == res.usage


# ============================================================================
# 6. Security & Credential Redaction Tests
# ============================================================================


def test_evaluator_rationale_redacts_credentials():
    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-sec", input="Task")
    dim = EvaluationDimension(name="execution_success", weight=1.0)

    # Failed result with secret in error string
    res = EvaluationResult(
        candidate_id="c-leak",
        model_id="m1",
        provider_id="p1",
        status="failed",
        error="Failed with API key sk-or-v1-secrettoken12345678 and Bearer supersecretbearer",
    )

    scores = evaluator.evaluate(task, [res], [dim])
    rationale = scores[0].rationale or ""

    assert "secrettoken12345678" not in rationale
    assert "supersecretbearer" not in rationale
    assert "[REDACTED]" in rationale


# ============================================================================
# 7. Architecture Boundary Tests
# ============================================================================


def test_evaluator_runs_without_network_or_api_keys(monkeypatch):
    """Verify evaluator has zero network, model, or provider credentials requirements."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    evaluator = DeterministicEvaluator()
    task = EvaluationTask(task_id="t-offline", input="Test", expected_output="OK")
    dim = EvaluationDimension(name="exact_match", weight=1.0)

    res = EvaluationResult(
        candidate_id="c-off",
        model_id="m-off",
        provider_id="p-off",
        status="success",
        output="OK",
    )

    summary = evaluator.evaluate_and_summarize(task, [res], [dim])
    assert summary.winner_candidate_id == "c-off"
    assert summary.scores[0].overall_score == 1.0
