from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.executor import MockModelExecutor, ModelExecutor
from app.local_executor import LocalModelBackend, LocalModelExecutor
from app.providers import (
    LocalModelProvider,
    ModelProvider,
    ProviderHealth,
    ProviderRegistry,
    ProviderStatus,
)
from app.registry import ModelRegistry
from app.routed_executor import RoutedModelExecutor
from app.schema import (
    ExecutionStatus,
    Modality,
    ModelExecutionRequest,
    ModelRequirements,
    ModelResult,
    RegisteredModel,
)
from app.validation import (
    DuplicateProviderError,
    ProviderError,
    ProviderNotFoundError,
)


def _sample_request(
    model_id: str = "qwen_2_5_coder_32b",
    input_text: str = "Write a quicksort implementation",
    expected_output: str = "Python quicksort with comments",
    metadata: dict[str, str] | None = None,
) -> ModelExecutionRequest:
    return ModelExecutionRequest(
        model_id=model_id,
        input=input_text,
        input_modalities=["text"],
        expected_output=expected_output,
        output_modalities=["text"],
        metadata=metadata or {},
    )


def test_provider_status_enum_values():
    assert ProviderStatus.HEALTHY.value == "healthy"
    assert ProviderStatus.DEGRADED.value == "degraded"
    assert ProviderStatus.UNHEALTHY.value == "unhealthy"


def test_provider_health_valid():
    health = ProviderHealth(
        provider_id="local",
        status=ProviderStatus.HEALTHY,
        latency_ms=1.5,
        message="Operational",
        metadata={"mode": "in_process"},
    )
    assert health.provider_id == "local"
    assert health.status == "healthy"
    assert health.latency_ms == 1.5
    assert health.metadata["mode"] == "in_process"


def test_provider_health_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ProviderHealth(
            provider_id="local",
            status="healthy",
            gpu_id="rtx_3090",  # Forbidden hardware field
        )

    with pytest.raises(ValidationError):
        ProviderHealth(
            provider_id="local",
            status="healthy",
            api_key="secret_token",  # Forbidden secret field
        )


def test_model_provider_subclassing():
    class IncompleteProvider(ModelProvider):
        provider_id = "incomplete"
        provider_type = "test"

    # Cannot instantiate without abstract methods
    with pytest.raises(TypeError):
        IncompleteProvider()


def test_provider_registry_register_and_get():
    registry = ProviderRegistry()
    provider = LocalModelProvider(provider_id="local_main")
    registry.register(provider)

    retrieved = registry.get("local_main")
    assert retrieved is provider
    assert retrieved.provider_id == "local_main"
    assert retrieved.provider_type == "local"


def test_provider_registry_rejects_duplicate():
    registry = ProviderRegistry()
    provider1 = LocalModelProvider(provider_id="local_main")
    provider2 = LocalModelProvider(provider_id="local_main")

    registry.register(provider1)
    with pytest.raises(DuplicateProviderError) as exc_info:
        registry.register(provider2)
    assert "already registered" in str(exc_info.value)


def test_provider_registry_missing_provider():
    registry = ProviderRegistry()
    with pytest.raises(ProviderNotFoundError) as exc_info:
        registry.get("non_existent_provider")
    assert "not found in registry" in str(exc_info.value)


def test_provider_registry_deterministic_listing():
    registry = ProviderRegistry()
    registry.register(LocalModelProvider(provider_id="zebra_provider"))
    registry.register(LocalModelProvider(provider_id="alpha_provider"))
    registry.register(LocalModelProvider(provider_id="beta_provider"))

    providers = registry.list()
    assert [p.provider_id for p in providers] == [
        "alpha_provider",
        "beta_provider",
        "zebra_provider",
    ]


def test_provider_registry_remove():
    registry = ProviderRegistry()
    registry.register(LocalModelProvider(provider_id="temp_provider"))
    assert len(registry.list()) == 1

    registry.remove("temp_provider")
    assert len(registry.list()) == 0

    with pytest.raises(ProviderNotFoundError):
        registry.remove("temp_provider")


def test_provider_registry_health_checks():
    registry = ProviderRegistry()
    p1 = LocalModelProvider(provider_id="p1", status=ProviderStatus.HEALTHY)
    p2 = LocalModelProvider(provider_id="p2", status=ProviderStatus.DEGRADED)
    registry.register(p1)
    registry.register(p2)

    h1 = registry.check_health("p1")
    assert h1.status == "healthy"

    all_health = registry.check_all_health()
    assert len(all_health) == 2
    assert all_health[0].provider_id == "p1"
    assert all_health[0].status == "healthy"
    assert all_health[1].provider_id == "p2"
    assert all_health[1].status == "degraded"


def test_local_model_provider_successful_execution():
    provider = LocalModelProvider(provider_id="local_engine")
    req = _sample_request()
    res = provider.execute(req)

    assert isinstance(res, ModelResult)
    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.model_id == req.model_id
    assert res.metadata["provider_id"] == "local_engine"
    assert res.metadata["provider_type"] == "local"
    assert "[local:" in res.output


def test_local_model_provider_health_check():
    provider = LocalModelProvider(provider_id="local_engine")
    health = provider.health_check()

    assert health.provider_id == "local_engine"
    assert health.status == "healthy"
    assert health.latency_ms == 0.0


def test_routed_model_executor_successful_execution():
    provider_registry = ProviderRegistry()
    local_provider = LocalModelProvider(provider_id="local")
    provider_registry.register(local_provider)

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        default_provider_id="local",
    )
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.model_id == req.model_id
    assert res.metadata["provider_id"] == "local"
    assert res.metadata["routed"] == "true"
    assert res.metadata["execution_mode"] == "local"


def test_routed_model_executor_via_metadata_override():
    provider_registry = ProviderRegistry()
    provider_registry.register(LocalModelProvider(provider_id="local_a"))
    provider_registry.register(LocalModelProvider(provider_id="local_b"))

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        default_provider_id="local_a",
    )

    req = _sample_request(metadata={"provider_id": "local_b"})
    res = executor.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.metadata["provider_id"] == "local_b"


def test_routed_model_executor_via_explicit_binding():
    provider_registry = ProviderRegistry()
    provider_registry.register(LocalModelProvider(provider_id="special_provider"))

    executor = RoutedModelExecutor(provider_registry=provider_registry)
    executor.bind_model("special_code_model", "special_provider")

    req = _sample_request(model_id="special_code_model")
    res = executor.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.metadata["provider_id"] == "special_provider"


def test_routed_model_executor_via_model_registry():
    provider_registry = ProviderRegistry()
    provider_registry.register(LocalModelProvider(provider_id="local_nim_mock"))

    model_registry = ModelRegistry()
    model_registry.register(
        RegisteredModel(
            model_id="deepseek_r1_local",
            provider_id="local_nim_mock",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=16384,
        )
    )

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        model_registry=model_registry,
    )

    req = _sample_request(model_id="deepseek_r1_local")
    res = executor.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.model_id == "deepseek_r1_local"
    assert res.metadata["provider_id"] == "local_nim_mock"


def test_routed_model_executor_via_model_supported_providers():
    provider_registry = ProviderRegistry()
    provider_registry.register(LocalModelProvider(provider_id="provider_two"))

    model_registry = ModelRegistry()
    model_registry.register(
        RegisteredModel(
            model_id="multi_provider_model",
            supported_providers=["provider_one", "provider_two"],
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=16384,
        )
    )

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        model_registry=model_registry,
    )

    req = _sample_request(model_id="multi_provider_model")
    res = executor.execute(req)

    # Resolves to provider_two because provider_one is not in provider_registry
    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.metadata["provider_id"] == "provider_two"


def test_routed_model_executor_unresolvable_provider():
    provider_registry = ProviderRegistry()
    executor = RoutedModelExecutor(provider_registry=provider_registry)

    req = _sample_request(model_id="unknown_model")
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "Cannot resolve provider" in res.error


def test_routed_model_executor_missing_provider_in_registry():
    provider_registry = ProviderRegistry()
    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        default_provider_id="non_existent_provider",
    )

    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "not found in ProviderRegistry" in res.error


def test_routed_model_executor_provider_failure():
    class FailingProvider(ModelProvider):
        provider_id = "failing"
        provider_type = "mock"

        def execute(self, request: ModelExecutionRequest) -> ModelResult:
            return ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality="text",
                error="Provider rate limit exceeded",
            )

        def health_check(self) -> ProviderHealth:
            return ProviderHealth(provider_id="failing", status=ProviderStatus.UNHEALTHY)

    provider_registry = ProviderRegistry()
    provider_registry.register(FailingProvider())

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        default_provider_id="failing",
    )
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert res.error == "Provider rate limit exceeded"


def test_routed_model_executor_provider_exception_containment():
    class CrashingProvider(ModelProvider):
        provider_id = "crashing"
        provider_type = "mock"

        def execute(self, request: ModelExecutionRequest) -> ModelResult:
            raise ConnectionResetError("Connection abruptly closed by peer")

        def health_check(self) -> ProviderHealth:
            return ProviderHealth(provider_id="crashing", status=ProviderStatus.UNHEALTHY)

    provider_registry = ProviderRegistry()
    provider_registry.register(CrashingProvider())

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        default_provider_id="crashing",
    )
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "ConnectionResetError: Connection abruptly closed by peer" in res.error
    assert "Traceback" not in res.error
    assert 'File "' not in res.error


def test_routed_model_executor_invalid_provider_result():
    class MalformedProvider(ModelProvider):
        provider_id = "malformed"
        provider_type = "mock"

        def execute(self, request: ModelExecutionRequest):
            return "string_instead_of_ModelResult"

        def health_check(self) -> ProviderHealth:
            return ProviderHealth(provider_id="malformed", status=ProviderStatus.HEALTHY)

    provider_registry = ProviderRegistry()
    provider_registry.register(MalformedProvider())

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        default_provider_id="malformed",
    )
    req = _sample_request()
    res = executor.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "expected ModelResult" in res.error


def test_routed_model_executor_invalid_request_rejection():
    provider_registry = ProviderRegistry()
    executor = RoutedModelExecutor(provider_registry=provider_registry)

    with pytest.raises(ValueError) as exc:
        executor.execute("not_a_request_object")
    assert "Invalid request type" in str(exc.value)


def test_mock_model_executor_independence():
    mock_executor = MockModelExecutor()
    req = _sample_request()
    res = mock_executor.execute(req)

    assert isinstance(res, ModelResult)
    assert res.status == ExecutionStatus.SUCCESS.value
    assert "[mock-output:" in res.output


def test_local_model_executor_independence():
    local_executor = LocalModelExecutor()
    req = _sample_request()
    res = local_executor.execute(req)

    assert isinstance(res, ModelResult)
    assert res.status == ExecutionStatus.SUCCESS.value
    assert "[local:" in res.output


def test_core_schemas_forbid_provider_specific_leakage():
    # ModelRequirements forbids provider fields
    with pytest.raises(ValidationError):
        ModelRequirements(
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            provider_name="openrouter",
        )

    # ModelExecutionRequest forbids provider fields
    with pytest.raises(ValidationError):
        ModelExecutionRequest(
            model_id="test",
            input="hello",
            input_modalities=["text"],
            expected_output="hi",
            output_modalities=["text"],
            cloud_provider="scaleway",
        )

    # ModelResult forbids provider fields
    with pytest.raises(ValidationError):
        ModelResult(
            model_id="test",
            status="success",
            output="hi",
            output_modality="text",
            gpu_type="h100",
        )


def test_registered_model_provider_association_without_false_defaults():
    # When unspecified, provider_id is None and supported_providers is empty
    model_abstract = RegisteredModel(
        model_id="abstract_reasoning_model",
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        maximum_context_tokens=32768,
    )
    assert model_abstract.provider_id is None
    assert model_abstract.supported_providers == []

    # Explicit single provider
    model_local = RegisteredModel(
        model_id="local_model",
        provider_id="local",
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        maximum_context_tokens=8192,
    )
    assert model_local.provider_id == "local"

    # Explicit multiple providers
    model_multi = RegisteredModel(
        model_id="flexible_model",
        supported_providers=["nim", "openrouter", "local"],
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        maximum_context_tokens=65536,
    )
    assert model_multi.supported_providers == ["nim", "openrouter", "local"]


def test_provider_resolution_metadata_override_is_targeted_override_only():
    """1. request.metadata['provider_id'] is an explicit execution override only."""
    provider_registry = ProviderRegistry()
    provider_registry.register(LocalModelProvider(provider_id="normal_provider"))
    provider_registry.register(LocalModelProvider(provider_id="override_provider"))

    model_registry = ModelRegistry()
    model_registry.register(
        RegisteredModel(
            model_id="standard_model",
            provider_id="normal_provider",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
        )
    )

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        model_registry=model_registry,
    )

    # Standard execution without metadata override uses normal resolution (normal_provider)
    req_normal = _sample_request(model_id="standard_model")
    res_normal = executor.execute(req_normal)
    assert res_normal.metadata["provider_id"] == "normal_provider"

    # Targeted execution with metadata override routes to override_provider
    req_override = _sample_request(
        model_id="standard_model",
        metadata={"provider_id": "override_provider"},
    )
    res_override = executor.execute(req_override)
    assert res_override.metadata["provider_id"] == "override_provider"


def test_provider_resolution_does_not_arbitrarily_select_among_multiple_providers():
    """2 & 3. Multiple supported providers resolve deterministically in declared order without arbitrary selection."""
    provider_registry = ProviderRegistry()
    provider_registry.register(LocalModelProvider(provider_id="first_choice"))
    provider_registry.register(LocalModelProvider(provider_id="second_choice"))

    model_registry = ModelRegistry()
    model_registry.register(
        RegisteredModel(
            model_id="ordered_model",
            supported_providers=["first_choice", "second_choice"],
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
        )
    )

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        model_registry=model_registry,
    )

    req = _sample_request(model_id="ordered_model")

    # Run 10 times to verify deterministic selection of first healthy candidate
    for _ in range(10):
        resolved = executor.resolve_provider_id(req)
        assert resolved == "first_choice"
        res = executor.execute(req)
        assert res.metadata["provider_id"] == "first_choice"


def test_unhealthy_provider_cannot_silently_become_selected():
    """4. An unavailable or unhealthy provider cannot silently become the selected provider unless explicitly overridden."""
    provider_registry = ProviderRegistry()
    unhealthy_provider = LocalModelProvider(
        provider_id="unhealthy_provider",
        status=ProviderStatus.UNHEALTHY,
        health_message="Cluster is unreachable",
    )
    healthy_provider = LocalModelProvider(
        provider_id="healthy_fallback",
        status=ProviderStatus.HEALTHY,
    )
    provider_registry.register(unhealthy_provider)
    provider_registry.register(healthy_provider)

    model_registry = ModelRegistry()
    model_registry.register(
        RegisteredModel(
            model_id="resilient_model",
            supported_providers=["unhealthy_provider", "healthy_fallback"],
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
        )
    )

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        model_registry=model_registry,
    )

    req = _sample_request(model_id="resilient_model")

    # Unhealthy provider is NOT silently selected; healthy_fallback is selected instead
    resolved = executor.resolve_provider_id(req)
    assert resolved == "healthy_fallback"
    res = executor.execute(req)
    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.metadata["provider_id"] == "healthy_fallback"

    # If ALL supported providers are unhealthy, it fails controlled rather than selecting an unhealthy provider
    all_unhealthy_model = RegisteredModel(
        model_id="all_unhealthy_model",
        supported_providers=["unhealthy_provider"],
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        maximum_context_tokens=8192,
    )
    model_registry.register(all_unhealthy_model)

    req_unhealthy = _sample_request(model_id="all_unhealthy_model")
    res_unhealthy = executor.execute(req_unhealthy)
    assert res_unhealthy.status == ExecutionStatus.FAILED.value
    assert "Cannot resolve provider" in res_unhealthy.error

    # Explicit targeted override permits executing against the specific provider
    req_explicit_target = _sample_request(
        model_id="resilient_model",
        metadata={"provider_id": "unhealthy_provider"},
    )
    resolved_explicit = executor.resolve_provider_id(req_explicit_target)
    assert resolved_explicit == "unhealthy_provider"


def test_provider_resolution_does_not_perform_capability_matching():
    """5. Provider resolution does NOT perform capability matching. ModelRouter remains responsible for model selection."""
    provider_registry = ProviderRegistry()
    provider_registry.register(LocalModelProvider(provider_id="local_exec"))

    # Model in registry has speech_to_text and vision capabilities
    model_registry = ModelRegistry()
    model_registry.register(
        RegisteredModel(
            model_id="multimodal_audio_model",
            provider_id="local_exec",
            capabilities=["speech_to_text", "vision"],
            input_modalities=["audio", "image"],
            output_modalities=["text"],
            maximum_context_tokens=16384,
        )
    )

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        model_registry=model_registry,
    )

    # RoutedModelExecutor does NOT reject this text request because it does not do capability matching.
    # It only dispatches the model_id that was already selected.
    req = _sample_request(model_id="multimodal_audio_model")
    res = executor.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.model_id == "multimodal_audio_model"
    assert res.metadata["provider_id"] == "local_exec"

    # Verify that capability matching remains solely in ModelRouter
    from app.router import ModelRouter
    from app.validation import NoCompatibleModelError
    router = ModelRouter(model_registry)
    text_reqs = ModelRequirements(
        capabilities=["code_generation"],  # Not supported by multimodal_audio_model
        input_modalities=["text"],
        output_modalities=["text"],
    )
    with pytest.raises(NoCompatibleModelError):
        router.select_model(text_reqs)
