from __future__ import annotations

from .schema import ModelRequirements, RegisteredModel
from .validation import DuplicateModelError, ModelNotFoundError, validate_registered_model


class ModelRegistry:
    """In-memory, provider-independent model registry."""

    def __init__(self) -> None:
        self._models: dict[str, RegisteredModel] = {}

    def register(self, model: RegisteredModel) -> None:
        """Register a new model profile. Rejects duplicate model_id."""
        validate_registered_model(model)
        if model.model_id in self._models:
            raise DuplicateModelError(f"Model '{model.model_id}' is already registered.")
        self._models[model.model_id] = model

    def remove(self, model_id: str) -> None:
        """Remove a model by model_id. Raises ModelNotFoundError if not present."""
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        del self._models[model_id]

    def get(self, model_id: str) -> RegisteredModel:
        """Retrieve a registered model by model_id."""
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        return self._models[model_id]

    def list(self) -> list[RegisteredModel]:
        """List all registered models deterministically (ordered by model_id)."""
        return [self._models[k] for k in sorted(self._models.keys())]

    def find_candidates(self, requirements: ModelRequirements) -> list[RegisteredModel]:
        """Find and return all compatible models sorted by priority (desc) and model_id (asc)."""
        candidates: list[RegisteredModel] = []

        req_caps = set(requirements.capabilities)
        req_inputs = set(requirements.input_modalities)
        req_outputs = set(requirements.output_modalities)

        for model in self._models.values():
            # 1. Filter out unavailable models
            if not model.available:
                continue

            # 2. Filter models that cannot satisfy every required capability
            if not req_caps.issubset(set(model.capabilities)):
                continue

            # 3. Filter models that cannot accept required input modalities
            if not req_inputs.issubset(set(model.input_modalities)):
                continue

            # 4. Filter models that cannot produce required output modalities
            if not req_outputs.issubset(set(model.output_modalities)):
                continue

            # 5. Filter insufficient context-window models
            if (
                requirements.minimum_context_tokens is not None
                and model.maximum_context_tokens < requirements.minimum_context_tokens
            ):
                continue

            # 6. Filter models that cannot satisfy structured_output
            if requirements.structured_output and not model.supports_structured_output:
                continue

            # 7. Filter models that cannot satisfy streaming
            if requirements.streaming and not model.supports_streaming:
                continue

            candidates.append(model)

        # Sort deterministically: highest priority first, then alphabetical model_id
        candidates.sort(key=lambda m: (-m.priority, m.model_id))
        return candidates
