"""Tests for Multi-Model Evaluation Laboratory (M3.9)."""

import pytest
from pydantic import ValidationError

from app.evaluation import (
    EvaluationCandidate,
    EvaluationExperiment,
    EvaluationResult,
    EvaluationRunner,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
    ExperimentRunResult,
    InvalidEvaluationCandidateError,
    InvalidEvaluationExperimentError,
    InvalidExperimentRunResultError,
    ModelLaboratory,
    validate_evaluation_experiment,
    validate_experiment_run_result,
)
from app.executor import ModelExecutor
from app.schema import ExecutionStatus, Modality, ModelExecutionRequest, ModelResult


class FakeModelExecutor(ModelExecutor):
    """Deterministic fake ModelExecutor for laboratory testing."""

    def __init__(
        self,
        responses: dict[str, ModelResult] | None = None,
        exceptions: dict[str, Exception] | None = None,
        default_output: str = "Fake laboratory output",
    ) -> None:
        self.responses = responses or {}
        self.exceptions = exceptions or {}
        self.default_output = default_output
        self.captured_requests: list[ModelExecutionRequest] = []

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        self.captured_requests.append(request)

        # 1. Trigger exceptions if configured
        if request.model_id in self.exceptions:
            raise self.exceptions[request.model_id]
        prov_id = request.metadata.get("provider_id", "")
        if prov_id in self.exceptions:
            raise self.exceptions[prov_id]

        # 2. Configured responses
        if request.model_id in self.responses:
            return self.responses[request.model_id]
        if prov_id in self.responses:
            return self.responses[prov_id]

        # 3. Default success result
        primary_out = (
            request.output_modalities[0]
            if request.output_modalities
            else Modality.TEXT.value
        )
        return ModelResult(
            model_id=request.model_id,
            status=ExecutionStatus.SUCCESS.value,
            output=f"{self.default_output} for {request.model_id}",
            output_modality=primary_out,
            usage={"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60},
            metadata={"captured_provider": prov_id},
        )


# ============================================================================
# 1. EvaluationExperiment Contract Tests
# ============================================================================


def test_experiment_contract_valid():
    task = EvaluationTask(
        task_id="task-invest",
        input="Analyze portfolio allocation scenario.",
        expected_output="Risk-weighted assessment.",
        structured_output=True,
    )
    candidates = [
        EvaluationCandidate(
            candidate_id="cand-1",
            model_id="deepseek/deepseek-chat",
            provider_id="openrouter",
        ),
        EvaluationCandidate(
            candidate_id="cand-2",
            model_id="meta-llama/llama-3.3-70b-instruct",
            provider_id="openrouter",
        ),
    ]

    exp = EvaluationExperiment(
        experiment_id="exp-invest-001",
        task=task,
        candidates=candidates,
        metadata={"domain": "finance", "benchmark": "asset-alloc-v1"},
    )

    assert exp.experiment_id == "exp-invest-001"
    assert exp.task.task_id == "task-invest"
    assert len(exp.candidates) == 2
    assert exp.metadata["domain"] == "finance"
    validate_evaluation_experiment(exp)


def test_experiment_empty_id_rejected():
    task = EvaluationTask(task_id="t1", input="Input")
    with pytest.raises(ValidationError):
        EvaluationExperiment(experiment_id="", task=task)

    with pytest.raises(ValidationError):
        EvaluationExperiment(experiment_id="   ", task=task)


def test_experiment_invalid_task_rejected():
    with pytest.raises(ValidationError):
        EvaluationExperiment(
            experiment_id="exp-1",
            task={"not_a_valid_task": True},  # type: ignore
        )


def test_experiment_duplicate_candidate_ids_rejected():
    task = EvaluationTask(task_id="t1", input="Input")
    candidates = [
        EvaluationCandidate(candidate_id="dup-id", model_id="m1", provider_id="p1"),
        EvaluationCandidate(candidate_id="dup-id", model_id="m2", provider_id="p2"),
    ]
    with pytest.raises(ValidationError) as exc_info:
        EvaluationExperiment(
            experiment_id="exp-dup",
            task=task,
            candidates=candidates,
        )
    assert "Duplicate candidate_id detected" in str(exc_info.value)


def test_experiment_candidate_variants():
    """Verify supporting same model across different providers and different models on same provider."""
    task = EvaluationTask(task_id="t-var", input="Variant test")

    # Same model / different providers
    cand_same_model_diff_provider = [
        EvaluationCandidate(candidate_id="c1", model_id="deepseek/deepseek-chat", provider_id="openrouter"),
        EvaluationCandidate(candidate_id="c2", model_id="deepseek/deepseek-chat", provider_id="nim"),
    ]
    exp1 = EvaluationExperiment(
        experiment_id="exp-same-model",
        task=task,
        candidates=cand_same_model_diff_provider,
    )
    assert len(exp1.candidates) == 2
    validate_evaluation_experiment(exp1)

    # Different models / same provider
    cand_diff_model_same_provider = [
        EvaluationCandidate(candidate_id="c3", model_id="model-alpha", provider_id="openrouter"),
        EvaluationCandidate(candidate_id="c4", model_id="model-beta", provider_id="openrouter"),
    ]
    exp2 = EvaluationExperiment(
        experiment_id="exp-diff-model",
        task=task,
        candidates=cand_diff_model_same_provider,
    )
    assert len(exp2.candidates) == 2
    validate_evaluation_experiment(exp2)


def test_experiment_forbids_unexpected_fields():
    task = EvaluationTask(task_id="t1", input="Input")
    with pytest.raises(ValidationError):
        EvaluationExperiment(
            experiment_id="exp-1",
            task=task,
            extra_field="disallowed",
        )

    with pytest.raises(ValidationError):
        ExperimentRunResult(
            experiment_id="exp-1",
            summary=EvaluationSummary(task_id="t1"),
            unexpected_field="disallowed",
        )


# ============================================================================
# 2. ModelLaboratory Execution & Same Task Guarantee Tests
# ============================================================================


def test_laboratory_one_candidate_experiment():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(
        task_id="t-single",
        input="Write hello world in C",
        expected_output="printf(\"Hello, World!\\n\");",
    )
    candidate = EvaluationCandidate(
        candidate_id="cand-c",
        model_id="qwen/qwen-coder",
        provider_id="openrouter",
    )
    exp = EvaluationExperiment(
        experiment_id="exp-single-1",
        task=task,
        candidates=[candidate],
        metadata={"purpose": "demo"},
    )

    run_result = lab.run(exp)

    assert isinstance(run_result, ExperimentRunResult)
    assert run_result.experiment_id == "exp-single-1"
    assert run_result.summary.task_id == "t-single"
    assert len(run_result.summary.results) == 1
    assert run_result.metadata["purpose"] == "demo"
    validate_experiment_run_result(run_result)


def test_laboratory_multi_candidate_runner_delegation():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(task_id="t-multi", input="Solve 2x + 4 = 10")
    candidates = [
        EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1"),
        EvaluationCandidate(candidate_id="c2", model_id="m2", provider_id="p2"),
        EvaluationCandidate(candidate_id="c3", model_id="m3", provider_id="p3"),
    ]
    exp = EvaluationExperiment(
        experiment_id="exp-multi-1",
        task=task,
        candidates=candidates,
    )

    run_result = lab.run(exp)

    assert run_result.experiment_id == "exp-multi-1"
    assert len(run_result.summary.results) == 3
    assert len(fake_executor.captured_requests) == 3


def test_laboratory_same_task_guarantee():
    """Verify all candidates receive the EXACT same task specifications."""
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(
        task_id="task-strict-check",
        input="Analyze macroeconomic impact of interest rate hike.",
        expected_output="Four specific bullet points assessing equities, bonds, FX, and commodities.",
        input_modalities=["text"],
        output_modalities=["text"],
        structured_output=True,
        metadata={"priority": "high"},
    )
    candidates = [
        EvaluationCandidate(candidate_id="cand-a", model_id="model-a", provider_id="p-1"),
        EvaluationCandidate(candidate_id="cand-b", model_id="model-b", provider_id="p-1"),
        EvaluationCandidate(candidate_id="cand-c", model_id="model-a", provider_id="p-2"),
    ]
    exp = EvaluationExperiment(experiment_id="exp-same-task", task=task, candidates=candidates)

    lab.run(exp)

    assert len(fake_executor.captured_requests) == 3
    for req in fake_executor.captured_requests:
        assert req.input == "Analyze macroeconomic impact of interest rate hike."
        assert req.expected_output == "Four specific bullet points assessing equities, bonds, FX, and commodities."
        assert req.input_modalities == ["text"]
        assert req.output_modalities == ["text"]
        assert req.structured_output is True


def test_laboratory_execution_order_preserved():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(task_id="t-order", input="Order check")
    candidates = [
        EvaluationCandidate(candidate_id="c-omega", model_id="m-omega", provider_id="p1"),
        EvaluationCandidate(candidate_id="c-alpha", model_id="m-alpha", provider_id="p1"),
        EvaluationCandidate(candidate_id="c-beta", model_id="m-beta", provider_id="p1"),
    ]
    exp = EvaluationExperiment(experiment_id="exp-order", task=task, candidates=candidates)

    run_result = lab.run(exp)

    # Order must match supplied candidates exactly
    assert [r.candidate_id for r in run_result.summary.results] == ["c-omega", "c-alpha", "c-beta"]
    assert [req.model_id for req in fake_executor.captured_requests] == ["m-omega", "m-alpha", "m-beta"]


# ============================================================================
# 3. Result Integrity & Failure Isolation Tests
# ============================================================================


def test_laboratory_result_integrity():
    configured_res = ModelResult(
        model_id="deepseek/deepseek-chat",
        status=ExecutionStatus.SUCCESS.value,
        output="Detailed market commentary.",
        output_modality=Modality.TEXT.value,
        usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        metadata={"engine": "inference-v2"},
    )
    fake_executor = FakeModelExecutor(responses={"deepseek/deepseek-chat": configured_res})
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(task_id="t-integrity", input="Market query")
    candidate = EvaluationCandidate(
        candidate_id="cand-deepseek",
        model_id="deepseek/deepseek-chat",
        provider_id="openrouter",
        metadata={"track": "prod"},
    )
    exp = EvaluationExperiment(
        experiment_id="exp-integrity",
        task=task,
        candidates=[candidate],
        metadata={"exp_meta": "yes"},
    )

    run_result = lab.run(exp)
    res = run_result.summary.results[0]

    assert run_result.experiment_id == "exp-integrity"
    assert run_result.summary.task_id == "t-integrity"
    assert res.candidate_id == "cand-deepseek"
    assert res.model_id == "deepseek/deepseek-chat"
    assert res.provider_id == "openrouter"
    assert res.status == EvaluationStatus.SUCCESS.value
    assert res.output == "Detailed market commentary."
    assert res.latency_ms >= 0.0
    assert res.usage == {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
    assert res.metadata["engine"] == "inference-v2"


def test_laboratory_failure_isolation():
    """Verify Candidate A success, Candidate B failure, Candidate C success."""
    fake_executor = FakeModelExecutor(
        exceptions={"model-b": RuntimeError("Service unavailable")}
    )
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(task_id="t-fail-iso", input="Failure isolation test")
    candidates = [
        EvaluationCandidate(candidate_id="cA", model_id="model-a", provider_id="p1"),
        EvaluationCandidate(candidate_id="cB", model_id="model-b", provider_id="p1"),
        EvaluationCandidate(candidate_id="cC", model_id="model-c", provider_id="p1"),
    ]
    exp = EvaluationExperiment(experiment_id="exp-iso", task=task, candidates=candidates)

    run_result = lab.run(exp)

    results = run_result.summary.results
    assert len(results) == 3
    assert results[0].candidate_id == "cA"
    assert results[0].status == EvaluationStatus.SUCCESS.value

    assert results[1].candidate_id == "cB"
    assert results[1].status == EvaluationStatus.FAILED.value
    assert "Service unavailable" in (results[1].error or "")

    assert results[2].candidate_id == "cC"
    assert results[2].status == EvaluationStatus.SUCCESS.value


# ============================================================================
# 4. Repeatable Execution & Immutability Tests
# ============================================================================


def test_laboratory_repeated_execution_immutability():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(task_id="t-repeat", input="Repeatability check")
    candidate = EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1")
    exp = EvaluationExperiment(
        experiment_id="exp-repeat",
        task=task,
        candidates=[candidate],
        metadata={"env": "test"},
    )

    # First run
    res1 = lab.run(exp)
    # Second run on the same experiment
    res2 = lab.run(exp)

    # Distinct result instances produced
    assert res1 is not res2
    assert res1.summary is not res2.summary
    assert res1.experiment_id == res2.experiment_id == "exp-repeat"

    # Verify input experiment, task, and candidates were not mutated
    assert exp.experiment_id == "exp-repeat"
    assert exp.task.task_id == "t-repeat"
    assert exp.task.input == "Repeatability check"
    assert len(exp.candidates) == 1
    assert exp.candidates[0].candidate_id == "c1"
    assert exp.metadata == {"env": "test"}


# ============================================================================
# 5. Quality, Provider, Model, and Compute Boundaries
# ============================================================================


def test_laboratory_quality_boundary():
    """Verify laboratory does not judge, score, or pick a winner."""
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(task_id="t-no-judge", input="Judge boundary")
    candidates = [
        EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1"),
        EvaluationCandidate(candidate_id="c2", model_id="m2", provider_id="p1"),
    ]
    exp = EvaluationExperiment(experiment_id="exp-no-judge", task=task, candidates=candidates)

    run_result = lab.run(exp)

    # Scores MUST remain empty
    assert run_result.summary.scores == []
    # Winner MUST remain None
    assert run_result.summary.winner_candidate_id is None


def test_laboratory_invalid_runner_type_raises():
    with pytest.raises(TypeError) as exc_info:
        ModelLaboratory(None)  # type: ignore
    assert "requires an EvaluationRunner instance" in str(exc_info.value)


def test_laboratory_offline_zero_credentials(monkeypatch):
    """Verify laboratory runs without any internet access or API credentials."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)
    lab = ModelLaboratory(runner)

    task = EvaluationTask(task_id="t-offline-lab", input="Offline lab test")
    candidate = EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1")
    exp = EvaluationExperiment(experiment_id="exp-offline", task=task, candidates=[candidate])

    run_result = lab.run(exp)
    assert run_result.experiment_id == "exp-offline"
    assert len(run_result.summary.results) == 1
