from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import ExecutionStatus, Modality, ModelExecutionRequest, ModelResult
from .validation import validate_execution_request, validate_model_result


class ModelExecutor(ABC):
    """Abstract base class for provider-independent model execution."""

    @abstractmethod
    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        """Execute the request against the selected model."""
        pass


class MockModelExecutor(ModelExecutor):
    """Deterministic mock model executor for testing and contract validation."""

    def __init__(self, should_fail: bool = False, failure_error: str = "Simulated execution failure") -> None:
        self.should_fail = should_fail
        self.failure_error = failure_error

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        """Deterministically execute a model request without external dependencies."""
        validate_execution_request(request)

        primary_out_modality = request.output_modalities[0] if request.output_modalities else Modality.TEXT.value
        tokens_in = len(request.input.split())

        if self.should_fail:
            result = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out_modality,
                usage={"prompt_tokens": tokens_in, "completion_tokens": 0, "total_tokens": tokens_in},
                metadata={"mock": "true", "mode": "simulated_failure"},
                error=self.failure_error,
            )
            return validate_model_result(result)

        mock_output = f"[mock-output:{request.model_id}] {request.expected_output}"
        tokens_out = len(mock_output.split())

        result = ModelResult(
            model_id=request.model_id,
            status=ExecutionStatus.SUCCESS.value,
            output=mock_output,
            output_modality=primary_out_modality,
            usage={
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
            metadata={
                "mock": "true",
                "structured_output": str(request.structured_output).lower(),
                "streaming": str(request.streaming).lower(),
            },
            error=None,
        )
        return validate_model_result(result)
