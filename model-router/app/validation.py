from .schema import ExecutionPlan, ModelRequirements


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
