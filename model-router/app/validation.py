from .schema import ExecutionPlan, ModelRequirements, RegisteredModel


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
