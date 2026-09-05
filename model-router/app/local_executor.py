from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .executor import ModelExecutor
from .schema import ExecutionStatus, Modality, ModelExecutionRequest, ModelResult
from .validation import validate_execution_request, validate_model_result


def _sanitize_error_message(exc: Exception) -> str:
    """Format an exception into a concise, meaningful string without stack traces."""
    exc_type = type(exc).__name__
    msg = str(exc).strip()
    if not msg:
        return f"{exc_type}: local execution failed"
    first_line = msg.splitlines()[0].strip()
    return f"{exc_type}: {first_line}"


class LocalBackendResult(BaseModel):
    """Raw result returned by a local backend before adaptation to ModelResult."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(default=None, description="Generated content if successful.")
    output_modality: str = Field(default=Modality.TEXT.value, description="Output data modality.")
    usage: dict[str, int] = Field(default_factory=dict, description="Execution usage metrics.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Backend-specific metadata.")
    error: str | None = Field(default=None, description="Error message if execution failed at backend level.")

    @field_validator("output_modality")
    @classmethod
    def validate_output_modality(cls, v: str) -> str:
        allowed = {m.value for m in Modality}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid output modality '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()


class LocalModelBackend(ABC):
    """Abstract interface for local inference backends."""

    backend_type: str = "local"

    @abstractmethod
    def generate(self, request: ModelExecutionRequest) -> LocalBackendResult:
        """Execute local generation for the given request."""
        pass


class InProcessBackend(LocalModelBackend):
    """Deterministic in-process backend for proving local execution pipeline.

    NOTE: This is NOT an AI model. It provides synthetic deterministic output
    and synthetic token accounting for testing and contract verification.
    """

    backend_type: str = "in_process"

    def __init__(
        self,
        should_fail: bool = False,
        failure_error: str = "In-process execution failed",
        raise_on_generate: Exception | None = None,
    ) -> None:
        self.should_fail = should_fail
        self.failure_error = failure_error
        self.raise_on_generate = raise_on_generate

    def generate(self, request: ModelExecutionRequest) -> LocalBackendResult:
        """Deterministically produce a backend result for testing."""
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        # Synthetic token accounting for testing (not real model tokenization)
        tokens_in = len(request.input.split())

        primary_out = (
            request.output_modalities[0]
            if request.output_modalities
            else Modality.TEXT.value
        )

        if self.should_fail:
            return LocalBackendResult(
                content=None,
                output_modality=primary_out,
                usage={
                    "prompt_tokens": tokens_in,
                    "completion_tokens": 0,
                    "total_tokens": tokens_in,
                },
                metadata={
                    "backend_type": self.backend_type,
                    "execution_mode": "local",
                },
                error=self.failure_error,
            )

        content = f"[local:{request.model_id}] {request.expected_output}"
        tokens_out = len(content.split())

        backend_metadata: dict[str, str] = {
            "backend_type": self.backend_type,
            "execution_mode": "local",
        }
        if request.structured_output:
            backend_metadata["structured_output"] = "true"
        if request.streaming:
            backend_metadata["streaming"] = "true"

        return LocalBackendResult(
            content=content,
            output_modality=primary_out,
            usage={
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
            metadata=backend_metadata,
            error=None,
        )


DeterministicLocalBackend = InProcessBackend


class LocalModelExecutor(ModelExecutor):
    """Local execution adapter conforming to ModelExecutor contract."""

    def __init__(self, backend: LocalModelBackend | None = None) -> None:
        self.backend = backend if backend is not None else InProcessBackend()

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        """Execute the request via the configured local backend."""
        # 1. Validate request
        if not isinstance(request, ModelExecutionRequest):
            raise ValueError(f"Invalid request type: expected ModelExecutionRequest, got {type(request).__name__}")
        validate_execution_request(request)

        primary_out_modality = (
            request.output_modalities[0]
            if request.output_modalities
            else Modality.TEXT.value
        )
        backend_name = getattr(self.backend, "backend_type", "unknown")

        # 2. Invoke local backend with exception handling
        try:
            backend_res = self.backend.generate(request)
        except Exception as exc:
            err_msg = _sanitize_error_message(exc)
            failed_result = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out_modality,
                usage={
                    "prompt_tokens": len(request.input.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(request.input.split()),
                },
                metadata={
                    "backend_type": backend_name,
                    "execution_mode": "local",
                },
                error=err_msg,
            )
            return validate_model_result(failed_result)

        # 3. Handle invalid backend result type
        if not isinstance(backend_res, LocalBackendResult):
            invalid_type_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out_modality,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "backend_type": backend_name,
                    "execution_mode": "local",
                },
                error=f"Invalid backend result type: expected LocalBackendResult, got {type(backend_res).__name__}",
            )
            return validate_model_result(invalid_type_res)

        # 4. Handle backend-signaled failure
        if backend_res.error:
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=backend_res.output_modality or primary_out_modality,
                usage=backend_res.usage or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={
                    "backend_type": backend_name,
                    "execution_mode": "local",
                    **backend_res.metadata,
                },
                error=backend_res.error,
            )
            return validate_model_result(failed_res)

        # 5. Handle empty output
        if not backend_res.content or not backend_res.content.strip():
            empty_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=backend_res.output_modality or primary_out_modality,
                usage=backend_res.usage or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={
                    "backend_type": backend_name,
                    "execution_mode": "local",
                    **backend_res.metadata,
                },
                error="Backend produced empty output",
            )
            return validate_model_result(empty_res)

        # 6. Construct and validate successful ModelResult
        merged_metadata: dict[str, str] = {
            **request.metadata,
            "backend_type": backend_name,
            "execution_mode": "local",
            **backend_res.metadata,
        }

        try:
            success_result = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.SUCCESS.value,
                output=backend_res.content,
                output_modality=backend_res.output_modality,
                usage=backend_res.usage,
                metadata=merged_metadata,
                error=None,
            )
            return validate_model_result(success_result)
        except Exception as exc:
            err_msg = _sanitize_error_message(exc)
            fallback = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out_modality,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "backend_type": backend_name,
                    "execution_mode": "local",
                },
                error=f"Invalid backend output: {err_msg}",
            )
            return validate_model_result(fallback)
