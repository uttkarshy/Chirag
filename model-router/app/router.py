from __future__ import annotations

from .registry import ModelRegistry
from .schema import ModelRequirements, ModelSelection
from .validation import NoCompatibleModelError



class ModelRouter:
    """Deterministic, provider-independent model router."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry if registry is not None else ModelRegistry()

    def select_model(self, requirements: ModelRequirements) -> ModelSelection:
        """Select the highest-priority compatible model for the given requirements."""
        candidates = self.registry.find_candidates(requirements)

        if not candidates:
            # Diagnose reason for error message
            all_models = self.registry.list()
            if not all_models:
                raise NoCompatibleModelError("Registry is empty. No models are registered.")

            available_models = [m for m in all_models if m.available]
            if not available_models:
                raise NoCompatibleModelError("No models are currently available in the registry.")

            reasons: list[str] = []

            # Check capabilities
            req_caps = set(requirements.capabilities)
            cap_matches = [m for m in available_models if req_caps.issubset(set(m.capabilities))]
            if not cap_matches:
                supported_caps = sorted(list(set(c for m in available_models for c in m.capabilities)))
                reasons.append(
                    f"Required capabilities {requirements.capabilities} not satisfied. Available: {supported_caps}"
                )

            # Check input modalities
            req_inputs = set(requirements.input_modalities)
            input_matches = [m for m in available_models if req_inputs.issubset(set(m.input_modalities))]
            if not input_matches:
                reasons.append(f"Required input modalities {requirements.input_modalities} not supported.")

            # Check output modalities
            req_outputs = set(requirements.output_modalities)
            output_matches = [m for m in available_models if req_outputs.issubset(set(m.output_modalities))]
            if not output_matches:
                reasons.append(f"Required output modalities {requirements.output_modalities} not supported.")

            # Check context window
            if requirements.minimum_context_tokens is not None:
                context_matches = [
                    m for m in available_models
                    if m.maximum_context_tokens >= requirements.minimum_context_tokens
                ]
                if not context_matches:
                    max_available = max(m.maximum_context_tokens for m in available_models)
                    reasons.append(
                        f"Insufficient context window: requested {requirements.minimum_context_tokens}, max available is {max_available}."
                    )

            # Check structured output
            if requirements.structured_output:
                struct_matches = [m for m in available_models if m.supports_structured_output]
                if not struct_matches:
                    reasons.append("No available model supports structured output.")

            # Check streaming
            if requirements.streaming:
                stream_matches = [m for m in available_models if m.supports_streaming]
                if not stream_matches:
                    reasons.append("No available model supports streaming.")

            detail = "; ".join(reasons) if reasons else "No single model satisfies all required capabilities and constraints simultaneously."
            raise NoCompatibleModelError(f"No compatible model found: {detail}")

        best = candidates[0]
        matched_caps = sorted(list(set(requirements.capabilities) & set(best.capabilities)))
        reason = (
            f"Selected model '{best.model_id}' (priority {best.priority}) "
            f"satisfying capabilities: {requirements.capabilities}"
        )

        return ModelSelection(
            model_id=best.model_id,
            matched_capabilities=matched_caps,
            reason=reason,
            metadata={"priority": str(best.priority)},
        )
