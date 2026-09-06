from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelCapability(str, Enum):
    """Allowed abstract model capabilities for execution."""

    TEXT_GENERATION = "text_generation"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    CODE_GENERATION = "code_generation"


class Modality(str, Enum):
    """Allowed provider-independent data modalities."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ExecutionStatus(str, Enum):
    """Allowed status values for model execution."""

    SUCCESS = "success"
    FAILED = "failed"


class ProviderStatus(str, Enum):
    """Operational status of an inference provider adapter."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ProviderHealth(BaseModel):
    """Operational health assessment of an inference provider."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(..., description="Identifier of the provider evaluated.")
    status: str = Field(..., description="Health status: healthy, degraded, or unhealthy.")
    latency_ms: float | None = Field(default=None, ge=0.0, description="Observed round-trip latency in milliseconds.")
    message: str | None = Field(default=None, description="Diagnostic or descriptive operational status message.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Neutral operational metadata.")

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("provider_id cannot be empty.")
        return v.strip()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {s.value for s in ProviderStatus}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid provider status '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()


class PlanStep(BaseModel):
    """Provider-independent step in an execution plan."""

    id: str = Field(..., description="Step identifier within the plan.")
    title: str = Field(..., description="Short title of the step.")
    step_type: str = Field(..., description="Step type: inspect_source, reasoning, generation, verification.")
    inputs: list[str] = Field(default_factory=list, description="Input references or identifiers.")
    depends_on: list[str] = Field(default_factory=list, description="Dependencies on other steps.")
    requirements: list[str] = Field(default_factory=list, description="Specific requirements for this step.")
    expected_output: str = Field(..., description="Expected output description.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Optional provider-independent metadata.")


class ExecutionPlan(BaseModel):
    """Provider-independent execution plan."""

    plan_id: str = Field(..., description="Deterministic plan identifier.")
    goal: str = Field(..., description="The primary goal from the intent.")
    desired_output: str = Field(..., description="The expected final output.")
    steps: list[PlanStep] = Field(..., description="Ordered list of execution plan steps.")
    estimated_complexity: str = Field(default="simple", description="Complexity estimate.")
    requires_tools: bool = Field(default=False, description="Whether tools are required.")
    verification_criteria: list[str] = Field(default_factory=list, description="Verification criteria.")


class ModelRequirements(BaseModel):
    """Execution requirements for a model to fulfill a plan or step."""

    model_config = ConfigDict(extra="forbid")

    capabilities: list[str] = Field(..., description="Required model capabilities.")
    input_modalities: list[str] = Field(..., description="Input data modalities.")
    output_modalities: list[str] = Field(..., description="Output data modalities.")
    minimum_context_tokens: int | None = Field(
        default=None,
        description="Minimum context window tokens required, or None if unspecified.",
    )
    structured_output: bool = Field(default=False, description="Whether structured output is required.")
    streaming: bool = Field(default=False, description="Whether token/event streaming is required.")

    @field_validator("capabilities")
    @classmethod
    def validate_and_normalize_capabilities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Capabilities list cannot be empty.")

        allowed_caps = {c.value for c in ModelCapability}
        seen = set()
        normalized = []

        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Capability must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed_caps:
                raise ValueError(
                    f"Unknown capability '{item}'. Allowed: {sorted(allowed_caps)}"
                )
            if norm in seen:
                raise ValueError(f"Duplicate capability detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)

        return sorted(normalized)

    @field_validator("input_modalities")
    @classmethod
    def validate_and_normalize_input_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Input modalities list cannot be empty.")

        allowed_mods = {m.value for m in Modality}
        seen = set()
        normalized = []

        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Input modality must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed_mods:
                raise ValueError(
                    f"Invalid input modality '{item}'. Allowed: {sorted(allowed_mods)}"
                )
            if norm in seen:
                raise ValueError(f"Duplicate input modality detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)

        return sorted(normalized)

    @field_validator("output_modalities")
    @classmethod
    def validate_and_normalize_output_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Output modalities list cannot be empty.")

        allowed_mods = {m.value for m in Modality}
        seen = set()
        normalized = []

        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Output modality must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed_mods:
                raise ValueError(
                    f"Invalid output modality '{item}'. Allowed: {sorted(allowed_mods)}"
                )
            if norm in seen:
                raise ValueError(f"Duplicate output modality detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)

        return sorted(normalized)

    @field_validator("minimum_context_tokens")
    @classmethod
    def validate_minimum_context_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("minimum_context_tokens cannot be negative.")
        return v


class RegisteredModel(BaseModel):
    """Provider-independent profile of a registered model available for routing."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Unique, stable model identifier.")
    capabilities: list[str] = Field(..., description="Supported capabilities.")
    input_modalities: list[str] = Field(..., description="Accepted input modalities.")
    output_modalities: list[str] = Field(..., description="Produced output modalities.")
    maximum_context_tokens: int = Field(..., gt=0, description="Maximum context window tokens supported.")
    supports_structured_output: bool = Field(default=False, description="Whether structured output is supported.")
    supports_streaming: bool = Field(default=False, description="Whether token/event streaming is supported.")
    priority: int = Field(default=0, description="Selection priority score (higher values preferred).")
    available: bool = Field(default=True, description="Whether the model is currently available for routing.")
    provider_id: str | None = Field(
        default=None,
        description="Explicit provider identifier if this model profile is bound to a single provider.",
    )
    supported_providers: list[str] = Field(
        default_factory=list,
        description="List of provider identifiers capable of serving this model.",
    )
    metadata: dict[str, str] = Field(default_factory=dict, description="Provider-independent routing metadata.")

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_id must be a non-empty string.")
        return v.strip()

    @field_validator("capabilities")
    @classmethod
    def validate_and_normalize_capabilities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Capabilities list cannot be empty.")

        allowed_caps = {c.value for c in ModelCapability}
        seen = set()
        normalized = []

        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Capability must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed_caps:
                raise ValueError(
                    f"Unknown capability '{item}'. Allowed: {sorted(allowed_caps)}"
                )
            if norm in seen:
                raise ValueError(f"Duplicate capability detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)

        return sorted(normalized)

    @field_validator("input_modalities")
    @classmethod
    def validate_and_normalize_input_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Input modalities list cannot be empty.")

        allowed_mods = {m.value for m in Modality}
        seen = set()
        normalized = []

        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Input modality must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed_mods:
                raise ValueError(
                    f"Invalid input modality '{item}'. Allowed: {sorted(allowed_mods)}"
                )
            if norm in seen:
                raise ValueError(f"Duplicate input modality detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)

        return sorted(normalized)

    @field_validator("output_modalities")
    @classmethod
    def validate_and_normalize_output_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Output modalities list cannot be empty.")

        allowed_mods = {m.value for m in Modality}
        seen = set()
        normalized = []

        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Output modality must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed_mods:
                raise ValueError(
                    f"Invalid output modality '{item}'. Allowed: {sorted(allowed_mods)}"
                )
            if norm in seen:
                raise ValueError(f"Duplicate output modality detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)

        return sorted(normalized)

    @field_validator("maximum_context_tokens")
    @classmethod
    def validate_context_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("maximum_context_tokens must be positive.")
        return v

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, v: str | None) -> str | None:
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("provider_id must be a non-empty string when provided.")
            return v.strip()
        return None

    @field_validator("supported_providers")
    @classmethod
    def validate_supported_providers(cls, v: list[str]) -> list[str]:
        seen = set()
        normalized = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("provider identifier must be a non-empty string.")
            norm = item.strip()
            if norm in seen:
                raise ValueError(f"Duplicate provider identifier detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)
        return normalized


class ModelSelection(BaseModel):
    """Result of a deterministic model routing selection."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Identifier of the selected model.")
    matched_capabilities: list[str] = Field(..., description="Capabilities matched against requirements.")
    reason: str = Field(..., description="Deterministic explanation for the selection.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Provider-independent routing metadata.")


class ModelExecutionRequest(BaseModel):
    """Provider-independent request to execute a model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Unique identifier of the model to execute.")
    input: str = Field(..., description="Input prompt or content to be processed.")
    input_modalities: list[str] = Field(..., description="Input data modalities.")
    expected_output: str = Field(..., description="Description or specification of expected output.")
    output_modalities: list[str] = Field(..., description="Desired output data modalities.")
    structured_output: bool = Field(default=False, description="Whether output must conform to a structured schema.")
    streaming: bool = Field(default=False, description="Whether streaming delivery is requested.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Provider-independent execution metadata.")

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_id cannot be empty.")
        return v.strip()

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("input cannot be empty.")
        return v

    @field_validator("expected_output")
    @classmethod
    def validate_expected_output(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("expected_output cannot be empty.")
        return v

    @field_validator("input_modalities")
    @classmethod
    def validate_input_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("input_modalities cannot be empty.")
        allowed = {m.value for m in Modality}
        seen = set()
        normalized = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Input modality must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed:
                raise ValueError(f"Invalid input modality '{item}'. Allowed: {sorted(allowed)}")
            if norm in seen:
                raise ValueError(f"Duplicate input modality detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)
        return sorted(normalized)

    @field_validator("output_modalities")
    @classmethod
    def validate_output_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("output_modalities cannot be empty.")
        allowed = {m.value for m in Modality}
        seen = set()
        normalized = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Output modality must be a non-empty string.")
            norm = item.strip().lower()
            if norm not in allowed:
                raise ValueError(f"Invalid output modality '{item}'. Allowed: {sorted(allowed)}")
            if norm in seen:
                raise ValueError(f"Duplicate output modality detected: '{norm}'")
            seen.add(norm)
            normalized.append(norm)
        return sorted(normalized)


class ModelResult(BaseModel):
    """Provider-independent result of model execution."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Identifier of the model that produced this result.")
    status: str = Field(..., description="Execution status: success or failed.")
    output: str | None = Field(default=None, description="Produced output content if successful.")
    output_modality: str = Field(..., description="Canonical output modality of the result.")
    usage: dict[str, int] = Field(default_factory=dict, description="Usage metrics (e.g. token counts).")
    metadata: dict[str, str] = Field(default_factory=dict, description="Execution metadata.")
    error: str | None = Field(default=None, description="Error message if execution failed.")

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_id cannot be empty.")
        return v.strip()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {s.value for s in ExecutionStatus}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid execution status '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()

    @field_validator("output_modality")
    @classmethod
    def validate_output_modality(cls, v: str) -> str:
        allowed = {m.value for m in Modality}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid output modality '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()
