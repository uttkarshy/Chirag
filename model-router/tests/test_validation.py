import pytest
from pydantic import ValidationError

from app.schema import ExecutionPlan, ModelRequirements, PlanStep, RegisteredModel
from app.validation import (
    DuplicateModelError,
    ModelNotFoundError,
    NoCompatibleModelError,
    RoutingError,
    validate_execution_plan_for_routing,
    validate_model_requirements,
    validate_registered_model,
)


def test_invalid_capability_rejection():
    with pytest.raises(ValidationError) as exc_info:
        ModelRequirements(
            capabilities=["quantum_teleportation"],
            input_modalities=["text"],
            output_modalities=["text"],
        )
    assert "Unknown capability 'quantum_teleportation'" in str(exc_info.value)


def test_invalid_modality_rejection():
    with pytest.raises(ValidationError) as exc_info:
        ModelRequirements(
            capabilities=["text_generation"],
            input_modalities=["hologram"],
            output_modalities=["text"],
        )
    assert "Invalid input modality 'hologram'" in str(exc_info.value)


def test_negative_context_tokens_rejection():
    with pytest.raises(ValidationError) as exc_info:
        ModelRequirements(
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            minimum_context_tokens=-100,
        )
    assert "minimum_context_tokens cannot be negative" in str(exc_info.value)


def test_duplicate_capabilities_rejection():
    with pytest.raises(ValidationError) as exc_info:
        ModelRequirements(
            capabilities=["text_generation", "text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
        )
    assert "Duplicate capability detected" in str(exc_info.value)


def test_duplicate_modalities_rejection():
    with pytest.raises(ValidationError) as exc_info:
        ModelRequirements(
            capabilities=["text_generation"],
            input_modalities=["text", "text"],
            output_modalities=["text"],
        )
    assert "Duplicate input modality detected" in str(exc_info.value)


def test_empty_capabilities_rejection():
    with pytest.raises(ValidationError) as exc_info:
        ModelRequirements(
            capabilities=[],
            input_modalities=["text"],
            output_modalities=["text"],
        )
    assert "Capabilities list cannot be empty" in str(exc_info.value)


def test_empty_plan_steps_rejection():
    plan = ExecutionPlan(
        plan_id="plan_empty",
        goal="Do something",
        desired_output="An output",
        steps=[],
    )
    with pytest.raises(ValueError) as exc_info:
        validate_execution_plan_for_routing(plan)
    assert "must contain at least one step" in str(exc_info.value)


def test_validate_model_requirements_helper():
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    validated = validate_model_requirements(req)
    assert validated == req


def test_registered_model_zero_or_negative_context_rejection():
    with pytest.raises(ValidationError):
        RegisteredModel(
            model_id="zero_context",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=0,
        )


def test_registered_model_empty_model_id_rejection():
    with pytest.raises(ValidationError):
        RegisteredModel(
            model_id="",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
        )


def test_routing_exception_hierarchy():
    assert issubclass(NoCompatibleModelError, RoutingError)
    assert issubclass(DuplicateModelError, RoutingError)
    assert issubclass(ModelNotFoundError, RoutingError)
