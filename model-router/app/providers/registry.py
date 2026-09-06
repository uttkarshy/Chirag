from __future__ import annotations

from ..schema import ProviderHealth
from ..validation import DuplicateProviderError, ProviderNotFoundError
from .base import ModelProvider


class ProviderRegistry:
    """In-memory registry of inference provider adapters."""

    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}

    def register(self, provider: ModelProvider) -> None:
        """Register a new provider adapter. Rejects duplicate provider_id."""
        if not hasattr(provider, "provider_id") or not provider.provider_id or not provider.provider_id.strip():
            raise ValueError("provider_id cannot be empty.")
        if provider.provider_id in self._providers:
            raise DuplicateProviderError(f"Provider '{provider.provider_id}' is already registered.")
        self._providers[provider.provider_id] = provider

    def remove(self, provider_id: str) -> None:
        """Remove a provider by provider_id. Raises ProviderNotFoundError if missing."""
        if provider_id not in self._providers:
            raise ProviderNotFoundError(f"Provider '{provider_id}' not found in registry.")
        del self._providers[provider_id]

    def get(self, provider_id: str) -> ModelProvider:
        """Retrieve a registered provider by provider_id."""
        if provider_id not in self._providers:
            raise ProviderNotFoundError(f"Provider '{provider_id}' not found in registry.")
        return self._providers[provider_id]

    def list(self) -> list[ModelProvider]:
        """List all registered providers deterministically ordered by provider_id."""
        return [self._providers[k] for k in sorted(self._providers.keys())]

    def check_health(self, provider_id: str) -> ProviderHealth:
        """Check health of a specific registered provider."""
        provider = self.get(provider_id)
        return provider.health_check()

    def check_all_health(self) -> list[ProviderHealth]:
        """Check health of all registered providers deterministically."""
        return [self._providers[k].health_check() for k in sorted(self._providers.keys())]
