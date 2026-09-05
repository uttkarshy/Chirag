"""Chirag Model Router - Capability Contract and Deterministic Router (M3.2)."""

from .analyzer import analyze_execution_plan
from .registry import ModelRegistry
from .router import ModelRouter
from .schema import (
    ExecutionPlan,
    Modality,
    ModelCapability,
    ModelRequirements,
    ModelSelection,
    PlanStep,
    RegisteredModel,
)
from .validation import (
    DuplicateModelError,
    ModelNotFoundError,
    NoCompatibleModelError,
    RoutingError,
    validate_execution_plan_for_routing,
    validate_model_requirements,
    validate_registered_model,
)

__all__ = [
    "ModelCapability",
    "Modality",
    "ModelRequirements",
    "RegisteredModel",
    "ModelSelection",
    "PlanStep",
    "ExecutionPlan",
    "ModelRegistry",
    "ModelRouter",
    "RoutingError",
    "NoCompatibleModelError",
    "DuplicateModelError",
    "ModelNotFoundError",
    "analyze_execution_plan",
    "validate_model_requirements",
    "validate_execution_plan_for_routing",
    "validate_registered_model",
]
