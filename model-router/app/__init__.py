"""Chirag Model Router - Capability Contract, Router, Executor, and Providers (M3.5)."""

from .analyzer import analyze_execution_plan
from .executor import MockModelExecutor, ModelExecutor
from .local_executor import (
    DeterministicLocalBackend,
    InProcessBackend,
    LocalBackendResult,
    LocalModelBackend,
    LocalModelExecutor,
)
from .providers import (
    LocalModelProvider,
    ModelProvider,
    ProviderHealth,
    ProviderRegistry,
    ProviderStatus,
)
from .registry import ModelRegistry
from .routed_executor import RoutedModelExecutor
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
    DuplicateProviderError,
    ExecutionError,
    ModelNotFoundError,
    NoCompatibleModelError,
    ProviderError,
    ProviderExecutionError,
    ProviderNotFoundError,
    RoutingError,
    sanitize_error_message,
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
    "ProviderStatus",
    "ProviderHealth",
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
    "LocalModelExecutor",
    "LocalModelBackend",
    "InProcessBackend",
    "DeterministicLocalBackend",
    "LocalBackendResult",
    "ModelProvider",
    "ProviderRegistry",
    "LocalModelProvider",
    "RoutedModelExecutor",
    "RoutingError",
    "NoCompatibleModelError",
    "DuplicateModelError",
    "ModelNotFoundError",
    "ExecutionError",
    "ProviderError",
    "DuplicateProviderError",
    "ProviderNotFoundError",
    "ProviderExecutionError",
    "sanitize_error_message",
    "analyze_execution_plan",
    "validate_model_requirements",
    "validate_execution_plan_for_routing",
    "validate_registered_model",
    "validate_execution_request",
    "validate_model_result",
]
