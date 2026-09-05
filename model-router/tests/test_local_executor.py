import pytest
from pydantic import ValidationError

from app.executor import MockModelExecutor, ModelExecutor
from app.local_executor import (
    DeterministicLocalBackend,
    InProcessBackend,
    LocalBackendResult,
    LocalModelBackend,
    LocalModelExecutor,
)
from app.registry import ModelRegistry
from app.router import ModelRouter
from app.schema import (
    ExecutionStatus,
    Modality,
    ModelExecutionRequest,
    ModelRequirements,
    ModelResult,
    RegisteredModel,
)


def _sample_request(
    model_id: str = "local_qwen_fast",
    input_text: str = "Summarize the technical proposal",
    input_modalities: list[str] | None = None,
    expected_output: str = "Concise summary in bullet points",
    output_modalities: list[str] | None = None,
    structured_output: bool = False,
    streaming: bool = False,
    metadata: dict[str, str] | None = None,
) -> ModelExecutionRequest:
    return ModelExecutionRequest(
        model_id=model_id,
        input=input_text,
        input_modalities=input_modalities or ["text"],
        expected_output=expected_output,
        output_modalities=output_modalities or ["text"],
        structured_output=structured_output,
        streaming=streaming,
        metadata=metadata or {},
    )


def test_successful_local_execution():
    executor = LocalModelExecutor()
    req = _sample_request()
    res = executor.execute(req)

    assert isinstance(res, ModelResult)
    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.error is None
    assert res.output is not None


def test_correct_model_id_preserved():
    executor = LocalModelExecutor()
    req = _sample_request(model_id="chirag_local_llama_3")
    res = executor.execute(req)

    assert res.model_id == "chirag_local_llama_3"


def test_correct_output():
    executor = LocalModelExecutor()
    req = _sample_request(
        model_id="local_model_v1",
        expected_output="Deterministic verification output",
    )
    res = executor.execute(req)

    assert res.output == "[local:local_model_v1] Deterministic verification output"


def test_correct_output_modality():
    executor = LocalModelExecutor()
    req = _sample_request(
        output_modalities=["text"],
    )
    res = executor.execute(req)
    assert res.output_modality == Modality.TEXT.value

    # Multimodal output request where primary output modality is image
    req_multimodal = _sample_request(
        input_modalities=["image", "text"],
        output_modalities=["image", "text"],
    )
    res_multimodal = executor.execute(req_multimodal)
    assert res_multimodal.output_modality == Modality.IMAGE.value


def test_usage_accounting():
    executor = LocalModelExecutor()
    req = _sample_request(
        input_text="Four words in input",
        expected_output="Three words here",
    )
    res = executor.execute(req)

    assert "prompt_tokens" in res.usage
    assert "completion_tokens" in res.usage
    assert "total_tokens" in res.usage
    assert res.usage["prompt_tokens"] == 4
    assert res.usage["completion_tokens"] == len("[local:local_qwen_fast] Three words here".split())
    assert res.usage["total_tokens"] == res.usage["prompt_tokens"] + res.usage["completion_tokens"]


def test_metadata_preservation_and_enrichment():
    executor = LocalModelExecutor()
    req = _sample_request(
        metadata={"step_id": "step-01", "trace_id": "tr-99"},
        structured_output=True,
        streaming=True,
    )
    res = executor.execute(req)

    assert res.metadata.get("step_id") == "step-01"
    assert res.metadata.get("trace_id") == "tr-99"
    assert res.metadata.get("backend_type") == "in_process"
    assert res.metadata.get("execution_mode") == "local"
    assert res.metadata.get("structured_output") == "true"
    assert res.metadata.get("streaming") == "true"


def test_failed_backend_execution():
    backend = InProcessBackend(should_fail=True, failure_error="Simulated out of resources")
    executor = LocalModelExecutor(backend=backend)
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.model_id == req.model_id
    assert res.output is None
    assert res.error == "Simulated out of resources"
    assert res.usage["prompt_tokens"] == len(req.input.split())
    assert res.usage["completion_tokens"] == 0
    assert res.metadata.get("backend_type") == "in_process"
    assert res.metadata.get("execution_mode") == "local"


def test_backend_exception_handling():
    backend = InProcessBackend(raise_on_generate=RuntimeError("Local device driver crashed unexpectedly"))
    executor = LocalModelExecutor(backend=backend)
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.model_id == req.model_id
    assert res.output is None
    assert "RuntimeError: Local device driver crashed unexpectedly" in res.error
    # Must not contain traceback artifacts
    assert "Traceback" not in res.error
    assert 'File "' not in res.error


def test_invalid_request_rejection():
    executor = LocalModelExecutor()

    # Non-ModelExecutionRequest object
    with pytest.raises(ValueError) as exc:
        executor.execute("not a valid request object")
    assert "Invalid request type" in str(exc.value)

    # Empty model_id on ModelExecutionRequest
    with pytest.raises(ValidationError):
        ModelExecutionRequest(
            model_id="  ",
            input="Valid input",
            input_modalities=["text"],
            expected_output="Output",
            output_modalities=["text"],
        )


def test_invalid_backend_result_type():
    class RogueBackend(LocalModelBackend):
        backend_type = "rogue"

        def generate(self, request: ModelExecutionRequest):
            return "string instead of LocalBackendResult"

    executor = LocalModelExecutor(backend=RogueBackend())
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "Invalid backend result type" in res.error


def test_invalid_backend_result_empty_content():
    class EmptyBackend(LocalModelBackend):
        backend_type = "empty"

        def generate(self, request: ModelExecutionRequest):
            return LocalBackendResult(content="   ", output_modality="text")

    executor = LocalModelExecutor(backend=EmptyBackend())
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "empty output" in res.error.lower()


def test_invalid_backend_result_malformed_modality():
    class BadModalityBackend(LocalModelBackend):
        backend_type = "bad_modality"

        def generate(self, request: ModelExecutionRequest):
            # Attempt to return a dictionary or bypassed result with invalid modality
            # Pydantic validation on LocalBackendResult prevents bad modality at instantiation
            return LocalBackendResult.model_construct(
                content="Valid content",
                output_modality="telepathy_wave",
                usage={},
                metadata={},
                error=None,
            )

    executor = LocalModelExecutor(backend=BadModalityBackend())
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "Invalid backend output" in res.error or "Invalid output modality" in res.error


def test_deterministic_repeated_execution():
    executor = LocalModelExecutor()
    req = _sample_request()

    res1 = executor.execute(req)
    res2 = executor.execute(req)

    assert res1.model_dump() == res2.model_dump()


def test_executor_conforms_to_model_executor_interface():
    executor = LocalModelExecutor()
    assert isinstance(executor, ModelExecutor)
    assert hasattr(executor, "execute")
    assert callable(executor.execute)


def test_mock_model_executor_remains_independently_functional():
    mock_executor = MockModelExecutor()
    local_executor = LocalModelExecutor()
    req = _sample_request()

    mock_res = mock_executor.execute(req)
    local_res = local_executor.execute(req)

    assert isinstance(mock_res, ModelResult)
    assert isinstance(local_res, ModelResult)
    assert "[mock-output:" in mock_res.output
    assert "[local:" in local_res.output
    assert mock_res.status == ExecutionStatus.SUCCESS.value
    assert local_res.status == ExecutionStatus.SUCCESS.value


def test_custom_backend_injection():
    class CustomEchoBackend(LocalModelBackend):
        backend_type = "custom_echo"

        def generate(self, request: ModelExecutionRequest) -> LocalBackendResult:
            return LocalBackendResult(
                content=f"Echo: {request.input}",
                output_modality="text",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                metadata={"echo": "true"},
            )

    custom_backend = CustomEchoBackend()
    executor = LocalModelExecutor(backend=custom_backend)
    req = _sample_request(input_text="Ping")
    res = executor.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.output == "Echo: Ping"
    assert res.metadata["backend_type"] == "custom_echo"
    assert res.metadata["echo"] == "true"


def test_deterministic_local_backend_alias():
    assert DeterministicLocalBackend is InProcessBackend


def test_router_selection_to_local_executor_pipeline():
    registry = ModelRegistry()
    registry.register(
        RegisteredModel(
            model_id="local_expert_8b",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
            priority=50,
        )
    )
    router = ModelRouter(registry)

    # 1. Route based on requirements
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    assert selection.model_id == "local_expert_8b"

    # 2. Construct execution request from selection
    exec_req = ModelExecutionRequest(
        model_id=selection.model_id,
        input="Summarize the incident report",
        input_modalities=req.input_modalities,
        expected_output="Incident root cause summary",
        output_modalities=req.output_modalities,
    )

    # 3. Execute via LocalModelExecutor
    executor = LocalModelExecutor()
    result = executor.execute(exec_req)

    assert result.status == ExecutionStatus.SUCCESS.value
    assert result.model_id == "local_expert_8b"
    assert result.output == "[local:local_expert_8b] Incident root cause summary"
    assert result.metadata["backend_type"] == "in_process"
    assert result.metadata["execution_mode"] == "local"
