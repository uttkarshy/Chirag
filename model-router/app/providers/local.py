from __future__ import annotations

from ..local_executor import LocalModelExecutor
from ..schema import ModelExecutionRequest, ModelResult, ProviderHealth, ProviderStatus
from .base import ModelProvider


class LocalModelProvider(ModelProvider):
    """Inference provider adapter wrapping the local execution boundary (LocalModelExecutor)."""

    def __init__(
        self,
        provider_id: str = "local",
        executor: LocalModelExecutor | None = None,
        status: ProviderStatus | str = ProviderStatus.HEALTHY,
        health_message: str = "Local in-process provider operational",
    ) -> None:
        self.provider_id = provider_id
        self.provider_type = "local"
        self._executor = executor if executor is not None else LocalModelExecutor()
        self._status = status if isinstance(status, ProviderStatus) else ProviderStatus(status)
        self._health_message = health_message

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        """Delegate execution to the underlying LocalModelExecutor."""
        result = self._executor.execute(request)
        enriched_metadata = {
            **result.metadata,
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
        }
        return result.model_copy(update={"metadata": enriched_metadata})

    def health_check(self) -> ProviderHealth:
        """Return operational health assessment for this local provider."""
        return ProviderHealth(
            provider_id=self.provider_id,
            status=self._status,
            latency_ms=0.0,
            message=self._health_message,
            metadata={"provider_type": self.provider_type},
        )
