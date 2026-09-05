import json
import pytest
from pydantic import ValidationError

from app.executor import MockModelExecutor, ModelExecutor
from app.registry import ModelRegistry
from app.router import ModelRouter
from app.schema import (
    ExecutionStatus,
    Modality,
    ModelCapability,
    ModelExecutionRequest,
    ModelRequirements,
    ModelResult,
    RegisteredModel,
)
from app.validation import validate_execution_request, validate_model_result


def _sample_request(
    model_id: str = "local_text_fast",
    input_text: str = "Summarize quarterly results",
    input_modalities: list[str] | None = None,
    expected_output: str = "Quarterly summary in markdown",
    output_modalities: list[str] | None = None,
    structured_output: bool = False,
    streaming: bool = False,
) -> ModelExecutionRequest:
    return ModelExecutionRequest(
        model_id=model_id,
        input=input_text,
        input_modalities=input_modalities or ["text"],
        expected_output=expected_output,
        output_modalities=output_modalities or ["text"],
        structured_output=structured_output,
        streaming=streaming,
    )


def test_valid_model_execution_request():
    req = _sample_request()
    assert req.model_id == "local_text_fast"
    assert req.input == "Summarize quarterly results"
    assert req.input_modalities == ["text"]
    assert req.output_modalities == ["text"]
    assert req.structured_output is False
    assert req.streaming is False


def test_reject_empty_model_id():
    with pytest.raises(ValidationError) as exc_info:
        ModelExecutionRequest(
            model_id="",
            input="Hello",
            input_modalities=["text"],
            expected_output="Greeting",
            output_modalities=["text"],
        )
    assert "model_id cannot be empty" in str(exc_info.value)


def test_reject_empty_input():
    with pytest.raises(ValidationError) as exc_info:
        ModelExecutionRequest(
            model_id="local_model",
            input="   ",
            input_modalities=["text"],
            expected_output="Greeting",
            output_modalities=["text"],
        )
    assert "input cannot be empty" in str(exc_info.value)


def test_reject_invalid_input_modality():
    with pytest.raises(ValidationError) as exc_info:
        ModelExecutionRequest(
            model_id="local_model",
            input="Hello",
            input_modalities=["mind_reading"],
            expected_output="Greeting",
            output_modalities=["text"],
        )
    assert "Invalid input modality 'mind_reading'" in str(exc_info.value)


def test_reject_invalid_output_modality():
    with pytest.raises(ValidationError) as exc_info:
        ModelExecutionRequest(
            model_id="local_model",
            input="Hello",
            input_modalities=["text"],
            expected_output="Greeting",
            output_modalities=["hologram"],
        )
    assert "Invalid output modality 'hologram'" in str(exc_info.value)


def test_valid_model_result():
    res = ModelResult(
        model_id="local_text_fast",
        status="success",
        output="Result text",
        output_modality="text",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    assert res.model_id == "local_text_fast"
    assert res.status == "success"
    assert res.output == "Result text"
    assert res.output_modality == "text"


def test_model_result_invalid_status():
    with pytest.raises(ValidationError) as exc_info:
        ModelResult(
            model_id="local_model",
            status="pending",  # Invalid status
            output_modality="text",
        )
    assert "Invalid execution status 'pending'" in str(exc_info.value)


def test_successful_mock_model_executor_execution():
    executor = MockModelExecutor()
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == "success"
    assert res.model_id == "local_text_fast"
    assert res.output_modality == "text"
    assert "[mock-output:local_text_fast]" in res.output
    assert res.error is None
    assert res.usage["total_tokens"] > 0


def test_failed_execution_result():
    executor = MockModelExecutor(should_fail=True, failure_error="CUDA out of memory simulation")
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == "failed"
    assert res.model_id == "local_text_fast"
    assert res.output is None
    assert res.error == "CUDA out of memory simulation"


def test_executor_preserves_model_id():
    executor = MockModelExecutor()
    req = _sample_request(model_id="special_code_model_v2")
    res = executor.execute(req)

    assert res.model_id == "special_code_model_v2"


def test_executor_does_not_perform_routing():
    # Executor receives a specific model_id and executes it without checking alternative candidates
    executor = MockModelExecutor()
    req = _sample_request(model_id="assigned_model_123")
    res = executor.execute(req)

    # Must preserve assigned model_id and not mutate or route
    assert res.model_id == "assigned_model_123"
    assert not hasattr(executor, "select_model")
    assert not hasattr(executor, "find_candidates")


def test_router_remains_independent_of_executor():
    # Router does not know about or call executor
    registry = ModelRegistry()
    registry.register(
        RegisteredModel(
            model_id="router_only_model",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
            priority=100,
        )
    )
    router = ModelRouter(registry)

    # Router only returns ModelSelection, not ModelResult
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    assert selection.model_id == "router_only_model"
    assert not hasattr(router, "execute")
    assert not hasattr(selection, "output")


def test_request_and_result_are_serializable():
    req = _sample_request()
    json_str = req.model_dump_json()
    reconstructed_req = ModelExecutionRequest.model_validate_json(json_str)
    assert reconstructed_req == req

    executor = MockModelExecutor()
    res = executor.execute(req)
    res_json = res.model_dump_json()
    reconstructed_res = ModelResult.model_validate_json(res_json)
    assert reconstructed_res == res


def test_deterministic_mock_execution():
    executor = MockModelExecutor()
    req = _sample_request()

    res1 = executor.execute(req)
    res2 = executor.execute(req)

    assert res1.model_dump() == res2.model_dump()


def test_structured_output_request():
    executor = MockModelExecutor()
    req = _sample_request(
        expected_output="JSON array of validated user profiles",
        structured_output=True,
    )
    res = executor.execute(req)

    assert res.status == "success"
    assert res.metadata.get("structured_output") == "true"


def test_streaming_request():
    executor = MockModelExecutor()
    req = _sample_request(
        expected_output="Token stream",
        streaming=True,
    )
    res = executor.execute(req)

    assert res.status == "success"
    assert res.metadata.get("streaming") == "true"


def test_multimodal_request():
    executor = MockModelExecutor()
    req = _sample_request(
        input_text="Examine this diagram and transcribe voice notes",
        input_modalities=["audio", "image", "text"],
        expected_output="Annotated infographic image",
        output_modalities=["image", "text"],
    )
    res = executor.execute(req)

    assert res.status == "success"
    assert res.output_modality in ["image", "text"]


def test_no_provider_specific_fields_required_or_allowed():
    # Extra provider fields on ModelExecutionRequest are rejected
    with pytest.raises(ValidationError):
        ModelExecutionRequest(
            model_id="local_model",
            input="Test prompt",
            input_modalities=["text"],
            expected_output="Response",
            output_modalities=["text"],
            provider="openai",  # Forbidden
        )

    # Extra provider fields on ModelResult are rejected
    with pytest.raises(ValidationError):
        ModelResult(
            model_id="local_model",
            status="success",
            output="Done",
            output_modality="text",
            api_key="sk-secret",  # Forbidden
        )


def test_end_to_end_router_to_executor_boundary():
    # Demonstrate clean end-to-end handoff:
    # ModelRequirements -> ModelRouter -> ModelSelection -> ModelExecutionRequest -> ModelExecutor -> ModelResult
    registry = ModelRegistry()
    registry.register(
        RegisteredModel(
            model_id="code_expert_local",
            capabilities=["code_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=16384,
            priority=100,
        )
    )
    router = ModelRouter(registry)

    # Step 1: Route
    requirements = ModelRequirements(
        capabilities=["code_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        minimum_context_tokens=8192,
    )
    selection = router.select_model(requirements)
    assert selection.model_id == "code_expert_local"

    # Step 2: Build execution request
    execution_request = ModelExecutionRequest(
        model_id=selection.model_id,
        input="Write a binary search function in Python",
        input_modalities=requirements.input_modalities,
        expected_output="Python binary search function with doctests",
        output_modalities=requirements.output_modalities,
    )

    # Step 3: Execute
    executor = MockModelExecutor()
    result = executor.execute(execution_request)

    assert result.status == "success"
    assert result.model_id == "code_expert_local"
    assert "Python binary search function" in result.output
