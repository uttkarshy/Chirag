"""Tests for EvaluationRunner (M3.8)."""

import pytest

from app.evaluation import (
    EvaluationCandidate,
    EvaluationResult,
    EvaluationRunner,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
    InvalidEvaluationCandidateError,
)
from app.executor import ModelExecutor
from app.schema import ExecutionStatus, Modality, ModelExecutionRequest, ModelResult


class FakeModelExecutor(ModelExecutor):
    """Deterministic fake ModelExecutor for runner testing without network access."""

    def __init__(
        self,
        responses: dict[str, ModelResult] | None = None,
        exceptions: dict[str, Exception] | None = None,
        default_output: str = "Fake model output",
    ) -> None:
        self.responses = responses or {}
        self.exceptions = exceptions or {}
        self.default_output = default_output
        self.captured_requests: list[ModelExecutionRequest] = []

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        self.captured_requests.append(request)

        # 1. Exception triggers by model_id or provider_id
        if request.model_id in self.exceptions:
            raise self.exceptions[request.model_id]
        prov_id = request.metadata.get("provider_id", "")
        if prov_id in self.exceptions:
            raise self.exceptions[prov_id]

        # 2. Configured responses by model_id or provider_id
        if request.model_id in self.responses:
            return self.responses[request.model_id]
        if prov_id in self.responses:
            return self.responses[prov_id]

        # 3. Default successful ModelResult
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
            usage={"prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36},
            metadata={"source": "fake_executor", "provider_id": prov_id},
        )


# ============================================================================
# 1. Basic Execution Tests
# ============================================================================


def test_runner_single_candidate_success():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(
        task_id="task-1",
        input="Summarize financial market movements.",
        expected_output="Three bullet points.",
    )
    candidates = [
        EvaluationCandidate(
            candidate_id="cand-1",
            model_id="deepseek/deepseek-chat",
            provider_id="openrouter",
        )
    ]

    summary = runner.run(task, candidates)

    assert isinstance(summary, EvaluationSummary)
    assert summary.task_id == "task-1"
    assert len(summary.results) == 1

    res = summary.results[0]
    assert res.candidate_id == "cand-1"
    assert res.model_id == "deepseek/deepseek-chat"
    assert res.provider_id == "openrouter"
    assert res.status == EvaluationStatus.SUCCESS.value
    assert "Fake model output for deepseek/deepseek-chat" in (res.output or "")
    assert res.latency_ms >= 0.0
    assert summary.scores == []
    assert summary.winner_candidate_id is None


def test_runner_multiple_candidates_success():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="task-multi", input="Compare Python vs Rust")
    candidates = [
        EvaluationCandidate(
            candidate_id="c-deepseek",
            model_id="deepseek/deepseek-chat",
            provider_id="openrouter",
        ),
        EvaluationCandidate(
            candidate_id="c-llama",
            model_id="meta-llama/llama-3.3-70b-instruct",
            provider_id="openrouter",
        ),
        EvaluationCandidate(
            candidate_id="c-qwen",
            model_id="qwen/qwen-2.5-coder-32b-instruct",
            provider_id="openrouter",
        ),
    ]

    summary = runner.run(task, candidates)

    assert len(summary.results) == 3
    assert [r.candidate_id for r in summary.results] == ["c-deepseek", "c-llama", "c-qwen"]
    assert all(r.status == EvaluationStatus.SUCCESS.value for r in summary.results)
    assert len(fake_executor.captured_requests) == 3


# ============================================================================
# 2. Request Mapping Tests
# ============================================================================


def test_runner_request_mapping():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(
        task_id="task-mapping",
        input="Write an SQL query.",
        expected_output="SELECT * FROM table;",
        input_modalities=["text"],
        output_modalities=["text"],
        structured_output=True,
        metadata={"category": "database", "priority": "high"},
    )
    candidate = EvaluationCandidate(
        candidate_id="cand-sql",
        model_id="qwen/qwen-2.5-coder-32b-instruct",
        provider_id="openrouter",
        metadata={"temperature": "0.1", "top_p": "0.9"},
    )

    runner.run(task, [candidate])

    assert len(fake_executor.captured_requests) == 1
    req = fake_executor.captured_requests[0]

    assert req.input == "Write an SQL query."
    assert req.expected_output == "SELECT * FROM table;"
    assert req.input_modalities == ["text"]
    assert req.output_modalities == ["text"]
    assert req.structured_output is True
    assert req.model_id == "qwen/qwen-2.5-coder-32b-instruct"

    # Provider ID passed in metadata
    assert req.metadata["provider_id"] == "openrouter"
    assert req.metadata["task_id"] == "task-mapping"
    assert req.metadata["candidate_id"] == "cand-sql"
    assert req.metadata["category"] == "database"
    assert req.metadata["temperature"] == "0.1"


def test_runner_expected_output_fallback():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    # Task without expected_output
    task = EvaluationTask(task_id="task-no-exp", input="Generate poem")
    candidate = EvaluationCandidate(
        candidate_id="cand-poem",
        model_id="m1",
        provider_id="p1",
    )

    runner.run(task, [candidate])

    req = fake_executor.captured_requests[0]
    assert req.expected_output != ""
    assert isinstance(req.expected_output, str)


# ============================================================================
# 3. Result Mapping Tests
# ============================================================================


def test_runner_result_mapping_success():
    expected_output = "def fib(n): pass"
    configured_res = ModelResult(
        model_id="qwen/qwen-coder",
        status=ExecutionStatus.SUCCESS.value,
        output=expected_output,
        output_modality=Modality.TEXT.value,
        usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        metadata={"custom_flag": "alpha"},
    )
    fake_executor = FakeModelExecutor(responses={"qwen/qwen-coder": configured_res})
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="task-code", input="Fibonacci")
    candidate = EvaluationCandidate(
        candidate_id="c-qwen",
        model_id="qwen/qwen-coder",
        provider_id="openrouter",
    )

    summary = runner.run(task, [candidate])
    res = summary.results[0]

    assert res.output == expected_output
    assert res.usage == {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
    assert res.metadata["custom_flag"] == "alpha"
    assert res.status == EvaluationStatus.SUCCESS.value
    assert res.error is None


def test_runner_result_mapping_failed_model_result():
    configured_res = ModelResult(
        model_id="deepseek/deepseek-chat",
        status=ExecutionStatus.FAILED.value,
        output=None,
        output_modality=Modality.TEXT.value,
        error="OpenRouter rate limit exceeded",
    )
    fake_executor = FakeModelExecutor(responses={"deepseek/deepseek-chat": configured_res})
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="task-fail", input="Hello")
    candidate = EvaluationCandidate(
        candidate_id="c-fail",
        model_id="deepseek/deepseek-chat",
        provider_id="openrouter",
    )

    summary = runner.run(task, [candidate])
    res = summary.results[0]

    assert res.status == EvaluationStatus.FAILED.value
    assert res.error == "OpenRouter rate limit exceeded"
    assert res.output is None
    assert res.latency_ms >= 0.0


# ============================================================================
# 4. Failure & Exception Isolation Tests
# ============================================================================


def test_runner_failure_isolation():
    """Verify Candidate A success, Candidate B failed result, Candidate C success."""
    failed_res = ModelResult(
        model_id="meta-llama/llama-3.3-70b-instruct",
        status=ExecutionStatus.FAILED.value,
        output=None,
        output_modality=Modality.TEXT.value,
        error="Inference timeout",
    )
    fake_executor = FakeModelExecutor(
        responses={"meta-llama/llama-3.3-70b-instruct": failed_res}
    )
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-iso", input="Test isolation")
    candidates = [
        EvaluationCandidate(candidate_id="cA", model_id="m-a", provider_id="p1"),
        EvaluationCandidate(candidate_id="cB", model_id="meta-llama/llama-3.3-70b-instruct", provider_id="p1"),
        EvaluationCandidate(candidate_id="cC", model_id="m-c", provider_id="p1"),
    ]

    summary = runner.run(task, candidates)

    assert len(summary.results) == 3
    assert summary.results[0].candidate_id == "cA"
    assert summary.results[0].status == EvaluationStatus.SUCCESS.value

    assert summary.results[1].candidate_id == "cB"
    assert summary.results[1].status == EvaluationStatus.FAILED.value
    assert summary.results[1].error == "Inference timeout"

    assert summary.results[2].candidate_id == "cC"
    assert summary.results[2].status == EvaluationStatus.SUCCESS.value


def test_runner_exception_isolation():
    """Verify Candidate B raising unexpected Exception does not abort Candidate A or C."""
    fake_executor = FakeModelExecutor(
        exceptions={"model-broken": RuntimeError("Uncaught backend crash")}
    )
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-exc", input="Test exceptions")
    candidates = [
        EvaluationCandidate(candidate_id="cA", model_id="m-ok-1", provider_id="p1"),
        EvaluationCandidate(candidate_id="cB", model_id="model-broken", provider_id="p1"),
        EvaluationCandidate(candidate_id="cC", model_id="m-ok-2", provider_id="p1"),
    ]

    summary = runner.run(task, candidates)

    assert len(summary.results) == 3
    assert summary.results[0].status == EvaluationStatus.SUCCESS.value

    # cB captured as failed result with safe error string
    assert summary.results[1].candidate_id == "cB"
    assert summary.results[1].status == EvaluationStatus.FAILED.value
    assert "Uncaught backend crash" in (summary.results[1].error or "")
    assert summary.results[1].latency_ms >= 0.0

    assert summary.results[2].status == EvaluationStatus.SUCCESS.value


def test_runner_exception_credential_redaction():
    """Verify secrets or API keys in exception messages are redacted."""
    secret_err = RuntimeError(
        "Request failed for key sk-or-v1-supersecretkey123456789 and Bearer secrettokenabcdef"
    )
    fake_executor = FakeModelExecutor(exceptions={"secret-model": secret_err})
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-sec", input="Test security")
    candidate = EvaluationCandidate(candidate_id="cSec", model_id="secret-model", provider_id="p1")

    summary = runner.run(task, [candidate])
    res = summary.results[0]

    assert res.status == EvaluationStatus.FAILED.value
    err_str = res.error or ""
    assert "supersecretkey123456789" not in err_str
    assert "secrettokenabcdef" not in err_str
    assert "[REDACTED]" in err_str


# ============================================================================
# 5. Latency Measurement Tests
# ============================================================================


def test_runner_latency_is_numeric_and_non_negative():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-lat", input="Latency test")
    candidate = EvaluationCandidate(candidate_id="cLat", model_id="m1", provider_id="p1")

    summary = runner.run(task, [candidate])
    res = summary.results[0]

    assert isinstance(res.latency_ms, float)
    assert res.latency_ms >= 0.0


# ============================================================================
# 6. Candidate Ordering & Summary Tests
# ============================================================================


def test_runner_preserves_supplied_candidate_order():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-ord", input="Order test")
    # Deliberately non-alphabetical
    candidates = [
        EvaluationCandidate(candidate_id="cand-z", model_id="model-z", provider_id="p1"),
        EvaluationCandidate(candidate_id="cand-a", model_id="model-a", provider_id="p1"),
        EvaluationCandidate(candidate_id="cand-m", model_id="model-m", provider_id="p1"),
    ]

    summary = runner.run(task, candidates)

    assert [r.candidate_id for r in summary.results] == ["cand-z", "cand-a", "cand-m"]
    assert [req.model_id for req in fake_executor.captured_requests] == ["model-z", "model-a", "model-m"]


def test_runner_summary_contains_no_judgment():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-judge", input="No judge test")
    candidates = [
        EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1"),
        EvaluationCandidate(candidate_id="c2", model_id="m2", provider_id="p1"),
    ]

    summary = runner.run(task, candidates)

    # Runner MUST NOT score or pick a winner
    assert summary.scores == []
    assert summary.winner_candidate_id is None
    assert summary.metadata["candidate_count"] == "2"


# ============================================================================
# 7. Duplicate Candidate Validation Tests
# ============================================================================


def test_runner_rejects_duplicate_candidate_ids():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-dup", input="Duplicate test")
    candidates = [
        EvaluationCandidate(candidate_id="c-dup", model_id="m1", provider_id="p1"),
        EvaluationCandidate(candidate_id="c-dup", model_id="m2", provider_id="p2"),
    ]

    with pytest.raises(InvalidEvaluationCandidateError) as exc_info:
        runner.run(task, candidates)

    assert "Duplicate candidate_id detected: 'c-dup'" in str(exc_info.value)
    # Execution MUST NOT begin when input validation fails
    assert len(fake_executor.captured_requests) == 0


# ============================================================================
# 8. Empty Candidate List Tests
# ============================================================================


def test_runner_empty_candidate_list():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-empty", input="Empty test")
    summary = runner.run(task, [])

    assert summary.task_id == "t-empty"
    assert summary.results == []
    assert summary.scores == []
    assert summary.winner_candidate_id is None
    assert summary.metadata["candidate_count"] == "0"
    assert len(fake_executor.captured_requests) == 0


# ============================================================================
# 9. Provider Independence & Architectural Boundaries
# ============================================================================


def test_runner_requires_model_executor_instance():
    with pytest.raises(TypeError) as exc_info:
        EvaluationRunner(None)  # type: ignore
    assert "requires a ModelExecutor instance" in str(exc_info.value)


def test_runner_preserves_provider_identity():
    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-prov", input="Provider identity test")
    candidate = EvaluationCandidate(
        candidate_id="c-deepseek-openrouter",
        model_id="deepseek/deepseek-chat",
        provider_id="openrouter",
    )

    summary = runner.run(task, [candidate])
    res = summary.results[0]

    assert res.model_id == "deepseek/deepseek-chat"
    assert res.provider_id == "openrouter"


def test_runner_zero_network_dependency_and_no_keys(monkeypatch):
    """Verify runner runs without internet access or API credentials."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    fake_executor = FakeModelExecutor()
    runner = EvaluationRunner(fake_executor)

    task = EvaluationTask(task_id="t-offline", input="Offline test")
    candidate = EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="p1")

    summary = runner.run(task, [candidate])
    assert len(summary.results) == 1
    assert summary.results[0].status == EvaluationStatus.SUCCESS.value
