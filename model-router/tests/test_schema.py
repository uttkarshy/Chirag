import pytest
from pydantic import ValidationError

from app.schema import Modality, ModelCapability, ModelRequirements


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
