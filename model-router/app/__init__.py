"""Chirag Model Router - Capability Contract, Router, and Executor (M3.3)."""

from .analyzer import analyze_execution_plan
from .executor import MockModelExecutor, ModelExecutor
from .registry import ModelRegistry
from .router import ModelRouter
from .schema import (
    ExecutionPlan,
    ExecutionStatus,
    Modality,
    ModelCapability,
    ModelExecutionRequest,
    ModelRequirements,
    ModelResult,
    ModelSelection,
    PlanStep,
    RegisteredModel,
)
from .validation import (
    DuplicateModelError,
    ExecutionError,
    ModelNotFoundError,
    NoCompatibleModelError,
    RoutingError,
    validate_execution_plan_for_routing,
    validate_execution_request,
    validate_model_requirements,
    validate_model_result,
    validate_registered_model,
)

__all__ = [
    "ModelCapability",
    "Modality",
    "ExecutionStatus",
    "ModelRequirements",
    "RegisteredModel",
    "ModelSelection",
    "ModelExecutionRequest",
    "ModelResult",
    "PlanStep",
    "ExecutionPlan",
    "ModelRegistry",
    "ModelRouter",
    "ModelExecutor",
    "MockModelExecutor",
    "RoutingError",
    "NoCompatibleModelError",
    "DuplicateModelError",
    "ModelNotFoundError",
    "ExecutionError",
    "analyze_execution_plan",
    "validate_model_requirements",
    "validate_execution_plan_for_routing",
    "validate_registered_model",
    "validate_execution_request",
    "validate_model_result",
]
