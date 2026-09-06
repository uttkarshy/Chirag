from __future__ import annotations

from .schema import (
    ExecutionPlan,
    ExecutionStatus,
    ModelExecutionRequest,
    ModelRequirements,
    ModelResult,
    RegisteredModel,
)


class RoutingError(Exception):
    """Base domain exception for model routing failures."""
    pass


class NoCompatibleModelError(RoutingError):
    """Raised when no registered model satisfies the required capabilities and constraints."""
    pass


class DuplicateModelError(RoutingError):
    """Raised when attempting to register a model with an already existing model_id."""
    pass


class ModelNotFoundError(RoutingError):
    """Raised when looking up or removing a model_id that is not registered."""
    pass


class ExecutionError(Exception):
    """Base domain exception for model execution failures."""
    pass


class ProviderError(Exception):
    """Base domain exception for inference provider operations."""
    pass


class DuplicateProviderError(ProviderError):
    """Raised when registering an already registered provider_id."""
    pass


class ProviderNotFoundError(ProviderError):
    """Raised when looking up or removing a provider_id that is not registered."""
    pass


class ProviderExecutionError(ProviderError):
    """Raised when provider execution fails."""
    pass


def validate_model_requirements(requirements: ModelRequirements) -> ModelRequirements:
    """Validate that model requirements adhere to contract invariants."""
    if not requirements.capabilities:
        raise ValueError("ModelRequirements must specify at least one capability.")
    if not requirements.input_modalities:
        raise ValueError("ModelRequirements must specify at least one input modality.")
    if not requirements.output_modalities:
        raise ValueError("ModelRequirements must specify at least one output modality.")
    if (
        requirements.minimum_context_tokens is not None
        and requirements.minimum_context_tokens < 0
    ):
        raise ValueError("minimum_context_tokens cannot be negative.")
    return requirements


def validate_execution_plan_for_routing(plan: ExecutionPlan) -> None:
    """Verify that an ExecutionPlan has required structure for capability analysis."""
    if not plan.steps:
        raise ValueError("ExecutionPlan must contain at least one step for analysis.")
    if not plan.desired_output or not plan.desired_output.strip():
        raise ValueError("ExecutionPlan must specify a desired_output.")


def validate_registered_model(model: RegisteredModel) -> RegisteredModel:
    """Validate registered model properties."""
    if not model.model_id or not model.model_id.strip():
        raise ValueError("model_id cannot be empty.")
    if not model.capabilities:
        raise ValueError("model must have at least one capability.")
    if not model.input_modalities:
        raise ValueError("model must have at least one input modality.")
    if not model.output_modalities:
        raise ValueError("model must have at least one output modality.")
    if model.maximum_context_tokens <= 0:
        raise ValueError("maximum_context_tokens must be positive.")
    return model


def validate_execution_request(request: ModelExecutionRequest) -> ModelExecutionRequest:
    """Validate that a ModelExecutionRequest satisfies contract invariants."""
    if not request.model_id or not request.model_id.strip():
        raise ValueError("model_id cannot be empty.")
    if not request.input or not request.input.strip():
        raise ValueError("input cannot be empty.")
    if not request.expected_output or not request.expected_output.strip():
        raise ValueError("expected_output cannot be empty.")
    if not request.input_modalities:
        raise ValueError("input_modalities cannot be empty.")
    if not request.output_modalities:
        raise ValueError("output_modalities cannot be empty.")
    return request


def validate_model_result(result: ModelResult) -> ModelResult:
    """Validate that a ModelResult satisfies contract invariants."""
    if not result.model_id or not result.model_id.strip():
        raise ValueError("model_id cannot be empty.")
    if result.status == ExecutionStatus.SUCCESS.value and result.output is None:
        raise ValueError("Successful ModelResult must contain an output.")
    if result.status == ExecutionStatus.FAILED.value and not result.error:
        raise ValueError("Failed ModelResult must contain an error description.")
    return result


def sanitize_error_message(exc: Exception) -> str:
    """Format an exception into a concise, meaningful string without stack traces."""
    exc_type = type(exc).__name__
    msg = str(exc).strip()
    if not msg:
        return f"{exc_type}: execution failed"
    first_line = msg.splitlines()[0].strip()
    return f"{exc_type}: {first_line}"
