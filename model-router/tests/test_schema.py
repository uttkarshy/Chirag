import pytest
from pydantic import ValidationError

from app.schema import Modality, ModelCapability, ModelRequirements, ModelSelection, RegisteredModel


def test_model_capability_enum_values():
    expected = {
        "text_generation",
        "vision",
        "image_generation",
        "video_generation",
        "speech_to_text",
        "text_to_speech",
        "code_generation",
    }
    actual = {c.value for c in ModelCapability}
    assert actual == expected


def test_modality_enum_values():
    expected = {"text", "image", "video", "audio"}
    actual = {m.value for m in Modality}
    assert actual == expected


def test_valid_model_requirements():
    req = ModelRequirements(
        capabilities=["text_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        minimum_context_tokens=4096,
        structured_output=True,
        streaming=True,
    )
    assert req.capabilities == ["text_generation"]
    assert req.input_modalities == ["text"]
    assert req.output_modalities == ["text"]
    assert req.minimum_context_tokens == 4096
    assert req.structured_output is True
    assert req.streaming is True


def test_extra_provider_fields_are_rejected():
    with pytest.raises(ValidationError):
        ModelRequirements(
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            provider="openai",  # Forbidden provider field
        )


def test_extra_model_fields_are_rejected():
    with pytest.raises(ValidationError):
        ModelRequirements(
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            model="gpt-4o",  # Forbidden model field
        )


def test_extra_gpu_fields_are_rejected():
    with pytest.raises(ValidationError):
        ModelRequirements(
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            gpu_memory="24GB",  # Forbidden hardware field
        )


def test_valid_registered_model():
    model = RegisteredModel(
        model_id="text_model_fast",
        capabilities=["text_generation", "code_generation"],
        input_modalities=["text"],
        output_modalities=["text"],
        maximum_context_tokens=32768,
        supports_structured_output=True,
        supports_streaming=True,
        priority=100,
        available=True,
    )
    assert model.model_id == "text_model_fast"
    assert model.capabilities == ["code_generation", "text_generation"]
    assert model.priority == 100
    assert model.available is True


def test_registered_model_rejects_extra_provider_fields():
    with pytest.raises(ValidationError):
        RegisteredModel(
            model_id="invalid_model",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
            provider="openai",  # Forbidden provider field
        )


def test_registered_model_rejects_extra_api_key_field():
    with pytest.raises(ValidationError):
        RegisteredModel(
            model_id="invalid_model",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=8192,
            api_key="secret",  # Forbidden secret field
        )


def test_valid_model_selection():
    sel = ModelSelection(
        model_id="local_model_a",
        matched_capabilities=["text_generation"],
        reason="Selected model with highest priority",
    )
    assert sel.model_id == "local_model_a"
    assert sel.matched_capabilities == ["text_generation"]


def test_model_selection_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ModelSelection(
            model_id="local_model_a",
            matched_capabilities=["text_generation"],
            reason="Selected",
            provider="anthropic",  # Forbidden provider field
        )
