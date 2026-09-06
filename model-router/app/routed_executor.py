from __future__ import annotations

from .executor import ModelExecutor
from .providers.base import ModelProvider
from .providers.registry import ProviderRegistry
from .registry import ModelRegistry
from .schema import ExecutionStatus, Modality, ModelExecutionRequest, ModelResult, ProviderStatus
from .validation import (
    ModelNotFoundError,
    ProviderNotFoundError,
    sanitize_error_message,
    validate_execution_request,
    validate_model_result,
)


class RoutedModelExecutor(ModelExecutor):
    """ModelExecutor that resolves and dispatches requests to registered providers."""

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        model_registry: ModelRegistry | None = None,
        bindings: dict[str, str] | None = None,
        default_provider_id: str | None = None,
    ) -> None:
        self.provider_registry = provider_registry
        self.model_registry = model_registry
        self._bindings: dict[str, str] = dict(bindings) if bindings else {}
        self.default_provider_id = default_provider_id

    def bind_model(self, model_id: str, provider_id: str) -> None:
        """Explicitly associate a model_id with a provider_id."""
        if not model_id or not model_id.strip():
            raise ValueError("model_id cannot be empty.")
        if not provider_id or not provider_id.strip():
            raise ValueError("provider_id cannot be empty.")
        self._bindings[model_id.strip()] = provider_id.strip()

    def resolve_provider_id(self, request: ModelExecutionRequest) -> str | None:
        """Deterministically resolve which provider should execute the request."""
        # 1. Explicit request metadata override (execution-time targeted override only)
        if "provider_id" in request.metadata and request.metadata["provider_id"].strip():
            return request.metadata["provider_id"].strip()

        # 2. Explicit executor binding
        if request.model_id in self._bindings:
            return self._bindings[request.model_id]

        # 3. ModelRegistry lookup
        if self.model_registry is not None:
            try:
                model = self.model_registry.get(request.model_id)
                # If model has a bound provider_id
                if model.provider_id:
                    try:
                        provider = self.provider_registry.get(model.provider_id)
                        health = provider.health_check()
                        if str(health.status).lower() != ProviderStatus.UNHEALTHY.value:
                            return model.provider_id
                    except ProviderNotFoundError:
                        # Return bound provider so caller receives explicit ProviderNotFoundError
                        return model.provider_id

                # Deterministically evaluate supported_providers in declared order
                if model.supported_providers:
                    for pid in model.supported_providers:
                        try:
                            provider = self.provider_registry.get(pid)
                            health = provider.health_check()
                            if str(health.status).lower() != ProviderStatus.UNHEALTHY.value:
                                return pid
                        except ProviderNotFoundError:
                            continue
            except ModelNotFoundError:
                pass

        # 4. Configured fallback default
        if self.default_provider_id:
            return self.default_provider_id

        return None

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        """Execute the request against the resolved provider."""
        if not isinstance(request, ModelExecutionRequest):
            raise ValueError(f"Invalid request type: expected ModelExecutionRequest, got {type(request).__name__}")
        validate_execution_request(request)

        primary_out = (
            request.output_modalities[0]
            if request.output_modalities
            else Modality.TEXT.value
        )

        provider_id = self.resolve_provider_id(request)
        if not provider_id:
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={
                    "prompt_tokens": len(request.input.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(request.input.split()),
                },
                metadata={"execution_mode": "routed"},
                error=f"Cannot resolve provider for model '{request.model_id}'.",
            )
            return validate_model_result(failed_res)

        try:
            provider = self.provider_registry.get(provider_id)
        except ProviderNotFoundError:
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={
                    "prompt_tokens": len(request.input.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(request.input.split()),
                },
                metadata={"execution_mode": "routed", "resolved_provider_id": provider_id},
                error=f"Provider '{provider_id}' not found in ProviderRegistry.",
            )
            return validate_model_result(failed_res)

        try:
            result = provider.execute(request)
        except Exception as exc:
            err_msg = sanitize_error_message(exc)
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={
                    "prompt_tokens": len(request.input.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(request.input.split()),
                },
                metadata={"execution_mode": "routed", "provider_id": provider_id},
                error=f"Provider execution failure: {err_msg}",
            )
            return validate_model_result(failed_res)

        if not isinstance(result, ModelResult):
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={"execution_mode": "routed", "provider_id": provider_id},
                error=f"Provider returned invalid result type: expected ModelResult, got {type(result).__name__}.",
            )
            return validate_model_result(failed_res)

        try:
            validated = validate_model_result(result)
            enriched_metadata = {
                **validated.metadata,
                "routed": "true",
            }
            return validated.model_copy(update={"metadata": enriched_metadata})
        except Exception as exc:
            err_msg = sanitize_error_message(exc)
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={"execution_mode": "routed", "provider_id": provider_id},
                error=f"Invalid ModelResult produced by provider: {err_msg}",
            )
            return validate_model_result(failed_res)
