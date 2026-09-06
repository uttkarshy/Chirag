from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import ModelExecutionRequest, ModelResult, ProviderHealth


class ModelProvider(ABC):
    """Abstract interface for inference provider adapters."""

    provider_id: str
    provider_type: str

    @abstractmethod
    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        """Execute a model execution request against this provider."""
        pass

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Deterministically check operational readiness of this provider."""
        pass
