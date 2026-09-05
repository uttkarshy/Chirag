import pytest

from app.registry import ModelRegistry
from app.router import ModelRouter
from app.schema import ModelRequirements, RegisteredModel
from app.validation import NoCompatibleModelError


def _make_model(
    model_id: str,
    capabilities: list[str] | None = None,
    input_modalities: list[str] | None = None,
    output_modalities: list[str] | None = None,
    max_tokens: int = 8192,
    structured_output: bool = True,
    streaming: bool = True,
    priority: int = 50,
    available: bool = True,
) -> RegisteredModel:
    return RegisteredModel(
        model_id=model_id,
        capabilities=capabilities or ["text_generation"],
        input_modalities=input_modalities or ["text"],
        output_modalities=output_modalities or ["text"],
        maximum_context_tokens=max_tokens,
        supports_structured_output=structured_output,
        supports_streaming=streaming,
        priority=priority,
        available=available,
    )


def test_select_exact_capability_match():
    registry = ModelRegistry()
    registry.register(_make_model("text_model", capabilities=["text_generation"]))
    registry.register(_make_model("vision_model", capabilities=["vision"]))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["vision"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    assert selection.model_id == "vision_model"
    assert selection.matched_capabilities == ["vision"]


def test_select_highest_priority_compatible_model():
    registry = ModelRegistry()
    registry.register(_make_model("model_low", priority=10))
    registry.register(_make_model("model_high", priority=100))
    registry.register(_make_model("model_mid", priority=50))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    assert selection.model_id == "model_high"


def test_priority_tie_resolves_by_model_id_deterministically():
    registry = ModelRegistry()
    registry.register(_make_model("model_charlie", priority=100))
    registry.register(_make_model("model_alpha", priority=100))
    registry.register(_make_model("model_bravo", priority=100))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    # Priority is tied at 100, so alphabetical order 'model_alpha' wins
    assert selection.model_id == "model_alpha"


def test_reject_unavailable_model():
    registry = ModelRegistry()
    registry.register(_make_model("offline_model", priority=200, available=False))
    registry.register(_make_model("online_model", priority=50, available=True))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    assert selection.model_id == "online_model"


def test_reject_missing_capability():
    registry = ModelRegistry()
    registry.register(_make_model("text_only", capabilities=["text_generation"]))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["code_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    with pytest.raises(NoCompatibleModelError) as exc_info:
        router.select_model(req)
    assert "code_generation" in str(exc_info.value)


def test_reject_incompatible_input_modality():
    registry = ModelRegistry()
    registry.register(_make_model("text_input_only", input_modalities=["text"], capabilities=["vision"]))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["vision"],
        input_modalities=["image", "text"],
        output_modalities=["text"],
    )
    with pytest.raises(NoCompatibleModelError) as exc_info:
        router.select_model(req)
    assert "input modalities" in str(exc_info.value)


def test_reject_incompatible_output_modality():
    registry = ModelRegistry()
    registry.register(
        _make_model(
            "text_output_only",
            capabilities=["image_generation"],
            output_modalities=["text"],
        )
    )

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["image_generation"],
        input_modalities=["text"],
        output_modalities=["image"],
    )
    with pytest.raises(NoCompatibleModelError) as exc_info:
        router.select_model(req)
    assert "output modalities" in str(exc_info.value)


def test_reject_insufficient_context_window():
    registry = ModelRegistry()
    registry.register(_make_model("small_context", max_tokens=4096))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        minimum_context_tokens=16384,
    )
    with pytest.raises(NoCompatibleModelError) as exc_info:
        router.select_model(req)
    assert "Insufficient context window" in str(exc_info.value)


def test_reject_structured_output_requirement_when_unsupported():
    registry = ModelRegistry()
    registry.register(_make_model("no_json", structured_output=False))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        structured_output=True,
    )
    with pytest.raises(NoCompatibleModelError) as exc_info:
        router.select_model(req)
    assert "structured output" in str(exc_info.value)


def test_reject_streaming_requirement_when_unsupported():
    registry = ModelRegistry()
    registry.register(_make_model("no_stream", streaming=False))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        streaming=True,
    )
    with pytest.raises(NoCompatibleModelError) as exc_info:
        router.select_model(req)
    assert "streaming" in str(exc_info.value)


def test_multiple_required_capabilities():
    registry = ModelRegistry()
    registry.register(_make_model("multimodal_model", capabilities=["code_generation", "vision"], priority=50))
    registry.register(_make_model("vision_only", capabilities=["vision"], priority=100))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["code_generation", "vision"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    assert selection.model_id == "multimodal_model"
    assert selection.matched_capabilities == ["code_generation", "vision"]


def test_no_compatible_model_returns_controlled_routing_error():
    registry = ModelRegistry()
    router = ModelRouter(registry)  # empty registry

    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    with pytest.raises(NoCompatibleModelError) as exc_info:
        router.select_model(req)
    assert "Registry is empty" in str(exc_info.value)


def test_same_registry_same_requirements_always_produces_identical_selection():
    registry = ModelRegistry()
    for i in range(10):
        registry.register(_make_model(f"model_{i}", priority=50))

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )

    results = [router.select_model(req).model_id for _ in range(50)]
    assert all(r == results[0] for r in results)
    assert results[0] == "model_0"  # alphabetical winner among tied priorities


def test_provider_model_infrastructure_details_not_required_by_routing():
    # Verify models can be registered and routed without provider names, endpoints, or GPU metadata
    registry = ModelRegistry()
    model = RegisteredModel(
        model_id="local_model_pure",
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        maximum_context_tokens=8192,
        priority=100,
    )
    registry.register(model)

    router = ModelRouter(registry)
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
    )
    selection = router.select_model(req)
    assert selection.model_id == "local_model_pure"
    assert "provider" not in selection.model_dump()
    assert "endpoint" not in selection.model_dump()
