import pytest

from app.registry import ModelRegistry
from app.schema import ModelRequirements, RegisteredModel
from app.validation import DuplicateModelError, ModelNotFoundError


def _sample_model(model_id: str, priority: int = 50, available: bool = True, max_tokens: int = 8192) -> RegisteredModel:
    return RegisteredModel(
        model_id=model_id,
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        maximum_context_tokens=max_tokens,
        supports_structured_output=True,
        supports_streaming=True,
        priority=priority,
        available=available,
    )


def test_register_model():
    registry = ModelRegistry()
    model = _sample_model("model_alpha")
    registry.register(model)

    assert registry.get("model_alpha") == model
    assert len(registry.list()) == 1


def test_reject_duplicate_model_id():
    registry = ModelRegistry()
    model1 = _sample_model("model_alpha")
    model2 = _sample_model("model_alpha", priority=100)

    registry.register(model1)
    with pytest.raises(DuplicateModelError) as exc_info:
        registry.register(model2)
    assert "already registered" in str(exc_info.value)


def test_retrieve_model():
    registry = ModelRegistry()
    model = _sample_model("model_beta")
    registry.register(model)

    retrieved = registry.get("model_beta")
    assert retrieved.model_id == "model_beta"


def test_retrieve_unknown_model_raises():
    registry = ModelRegistry()
    with pytest.raises(ModelNotFoundError):
        registry.get("nonexistent")


def test_remove_model():
    registry = ModelRegistry()
    model = _sample_model("model_gamma")
    registry.register(model)
    assert len(registry.list()) == 1

    registry.remove("model_gamma")
    assert len(registry.list()) == 0

    with pytest.raises(ModelNotFoundError):
        registry.get("model_gamma")


def test_remove_unknown_model_raises():
    registry = ModelRegistry()
    with pytest.raises(ModelNotFoundError):
        registry.remove("nonexistent")


def test_list_models_deterministically():
    registry = ModelRegistry()
    registry.register(_sample_model("model_z"))
    registry.register(_sample_model("model_a"))
    registry.register(_sample_model("model_m"))

    listed = registry.list()
    assert [m.model_id for m in listed] == ["model_a", "model_m", "model_z"]


def test_candidate_filtering():
    registry = ModelRegistry()
    m1 = _sample_model("m1", priority=10, max_tokens=4096)
    m2 = _sample_model("m2", priority=100, max_tokens=16384)
    m3 = _sample_model("m3", priority=50, available=False)  # unavailable
    registry.register(m1)
    registry.register(m2)
    registry.register(m3)

    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        minimum_context_tokens=8192,
    )
    candidates = registry.find_candidates(req)
    assert len(candidates) == 1
    assert candidates[0].model_id == "m2"
