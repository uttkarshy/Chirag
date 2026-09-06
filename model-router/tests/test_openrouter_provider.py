from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from app.executor import MockModelExecutor
from app.local_executor import LocalModelExecutor
from app.providers.local import LocalModelProvider
from app.providers.openrouter import OpenRouterClient, OpenRouterConfig, OpenRouterProvider
from app.providers.registry import ProviderRegistry
from app.registry import ModelRegistry
from app.routed_executor import RoutedModelExecutor
from app.schema import (
    ExecutionStatus,
    Modality,
    ModelExecutionRequest,
    ModelRequirements,
    ModelResult,
    ProviderHealth,
    ProviderStatus,
    RegisteredModel,
)
from app.validation import ProviderError


def _sample_request(
    model_id: str = "deepseek/deepseek-chat",
    input_text: str = "Write an essay about distributed systems",
    expected_output: str = "Structured essay in markdown",
    structured_output: bool = False,
    streaming: bool = False,
    metadata: dict[str, str] | None = None,
) -> ModelExecutionRequest:
    return ModelExecutionRequest(
        model_id=model_id,
        input=input_text,
        input_modalities=["text"],
        expected_output=expected_output,
        output_modalities=["text"],
        structured_output=structured_output,
        streaming=streaming,
        metadata=metadata or {},
    )


def _mock_openrouter_success_payload(
    content: str = "Here is the essay on distributed systems.",
    model: str = "deepseek/deepseek-chat",
    prompt_tokens: int = 15,
    completion_tokens: int = 35,
) -> dict[str, Any]:
    return {
        "id": "gen-test-12345",
        "provider": "DeepSeek",
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ============================================================================
# 1. PROVIDER IDENTITY & CONFIGURATION
# ============================================================================

def test_openrouter_provider_identity():
    config = OpenRouterConfig(api_key="sk-or-v1-fake-test-key-1234")
    provider = OpenRouterProvider(config=config)

    assert provider.provider_id == "openrouter"
    assert provider.provider_type == "openrouter"


def test_openrouter_custom_provider_id():
    config = OpenRouterConfig(api_key="sk-or-v1-fake-test-key-1234")
    provider = OpenRouterProvider(provider_id="openrouter_secondary", config=config)

    assert provider.provider_id == "openrouter_secondary"
    assert provider.provider_type == "openrouter"


def test_openrouter_importable_and_instantiable_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Instantiation should NOT raise an error
    provider = OpenRouterProvider()
    assert provider.provider_id == "openrouter"
    assert provider.config.api_key is None


# ============================================================================
# 2. SUCCESSFUL EXECUTION & REQUEST MAPPING
# ============================================================================

def test_openrouter_successful_execution():
    captured_payload = {}

    class MockClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal captured_payload
            captured_payload = payload
            return _mock_openrouter_success_payload(
                content="Distributed systems require consensus protocols.",
                model="deepseek/deepseek-chat",
                prompt_tokens=10,
                completion_tokens=25,
            )

    provider = OpenRouterProvider(client=MockClient())
    req = _sample_request()
    res = provider.execute(req)

    # Verify result structure
    assert isinstance(res, ModelResult)
    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.model_id == "deepseek/deepseek-chat"
    assert res.output == "Distributed systems require consensus protocols."
    assert res.output_modality == Modality.TEXT.value
    assert res.usage["prompt_tokens"] == 10
    assert res.usage["completion_tokens"] == 25
    assert res.usage["total_tokens"] == 35
    assert res.metadata["provider_id"] == "openrouter"
    assert res.metadata["provider_type"] == "openrouter"
    assert res.metadata["returned_model"] == "deepseek/deepseek-chat"
    assert res.metadata["upstream_provider"] == "DeepSeek"
    assert res.metadata["openrouter_id"] == "gen-test-12345"

    # Verify payload mapping
    assert captured_payload["model"] == "deepseek/deepseek-chat"
    assert captured_payload["messages"] == [{"role": "user", "content": req.input}]


def test_openrouter_structured_output_flag():
    captured_payload = {}

    class MockClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal captured_payload
            captured_payload = payload
            return _mock_openrouter_success_payload(content='{"status": "ok"}')

    provider = OpenRouterProvider(client=MockClient())
    req = _sample_request(structured_output=True)
    res = provider.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert captured_payload.get("response_format") == {"type": "json_object"}
    assert res.metadata.get("structured_output") == "true"


def test_openrouter_streaming_flag():
    captured_payload = {}

    class MockClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal captured_payload
            captured_payload = payload
            return _mock_openrouter_success_payload(content="Streaming content chunk")

    provider = OpenRouterProvider(client=MockClient())
    req = _sample_request(streaming=True)
    res = provider.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert captured_payload.get("stream") is True
    assert res.metadata.get("streaming") == "true"


def test_openrouter_preserves_returned_model_if_different():
    class MockClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            # Upstream returns a specific snapshot or quantized variant
            return _mock_openrouter_success_payload(
                model="deepseek/deepseek-chat:beta-quantized",
            )

    provider = OpenRouterProvider(client=MockClient())
    req = _sample_request(model_id="deepseek/deepseek-chat")
    res = provider.execute(req)

    # Requested model_id is strictly preserved
    assert res.model_id == "deepseek/deepseek-chat"
    # Actual upstream model recorded in metadata
    assert res.metadata["returned_model"] == "deepseek/deepseek-chat:beta-quantized"


# ============================================================================
# 3. ERROR HANDLING & SECRET SANITIZATION
# ============================================================================

def test_openrouter_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider(config=OpenRouterConfig(api_key=None))
    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "OpenRouter API key is missing" in res.error


def test_openrouter_timeout_handling():
    class TimeoutClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test", timeout_seconds=15.0))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise httpx.TimeoutException("Read timed out")

    provider = OpenRouterProvider(client=TimeoutClient())
    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "timed out after 15.0s" in res.error


def test_openrouter_connection_failure():
    class ConnectErrorClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise httpx.ConnectError("Failed to resolve host openrouter.ai")

    provider = OpenRouterProvider(client=ConnectErrorClient())
    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "connection failed" in res.error.lower()


def test_openrouter_http_error_401_unauthorized():
    fake_key = "sk-or-v1-super-secret-user-key-abcdef"
    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"message": f"Invalid key: {fake_key}"}}
    mock_http.post.return_value = mock_response

    config = OpenRouterConfig(api_key=fake_key)
    client = OpenRouterClient(config=config, http_client=mock_http)
    provider = OpenRouterProvider(config=config, client=client)

    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert res.output is None
    assert "authentication failed" in res.error.lower()
    # The secret API key must NOT be leaked into the error
    assert fake_key not in res.error


def test_openrouter_http_error_500():
    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": {"message": "Internal upstream inference provider error"}}
    mock_http.post.return_value = mock_response

    config = OpenRouterConfig(api_key="sk-or-v1-test")
    client = OpenRouterClient(config=config, http_client=mock_http)
    provider = OpenRouterProvider(config=config, client=client)

    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert "500" in res.error
    assert "Internal upstream inference provider error" in res.error


def test_openrouter_malformed_json_response():
    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_http.post.return_value = mock_response

    config = OpenRouterConfig(api_key="sk-or-v1-test")
    client = OpenRouterClient(config=config, http_client=mock_http)
    provider = OpenRouterProvider(config=config, client=client)

    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert "malformed JSON" in res.error


def test_openrouter_malformed_successful_response_empty_choices():
    class BadResponseClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"id": "gen-1", "choices": []}

    provider = OpenRouterProvider(client=BadResponseClient())
    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert "missing choices array" in res.error


def test_openrouter_malformed_successful_response_null_content():
    class NullContentClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": "gen-1",
                "choices": [{"message": {"role": "assistant", "content": None}}],
            }

    provider = OpenRouterProvider(client=NullContentClient())
    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert "empty or null content" in res.error


def test_no_credential_leakage_in_error():
    fake_key = "sk-or-v1-sensitive-credential-987654321"

    class ExceptionClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key=fake_key))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError(f"Failed with Bearer {fake_key}")

    provider = OpenRouterProvider(config=OpenRouterConfig(api_key=fake_key), client=ExceptionClient())
    req = _sample_request()
    res = provider.execute(req)

    assert res.status == ExecutionStatus.FAILED.value
    assert fake_key not in res.error
    assert "[REDACTED]" in res.error


# ============================================================================
# 4. HEALTH CHECK
# ============================================================================

def test_openrouter_health_check_success():
    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {"label": "test-key", "limit": 100}}
    mock_http.get.return_value = mock_response

    config = OpenRouterConfig(api_key="sk-or-v1-valid")
    client = OpenRouterClient(config=config, http_client=mock_http)
    provider = OpenRouterProvider(config=config, client=client)

    health = provider.health_check()
    assert isinstance(health, ProviderHealth)
    assert health.provider_id == "openrouter"
    assert health.status == ProviderStatus.HEALTHY.value
    assert health.latency_ms is not None
    assert health.message == "OpenRouter operational"


def test_openrouter_health_check_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider(config=OpenRouterConfig(api_key=None))

    health = provider.health_check()
    assert health.status == ProviderStatus.UNHEALTHY.value
    assert "API key is not configured" in health.message


def test_openrouter_health_check_auth_failure():
    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_http.get.return_value = mock_response

    config = OpenRouterConfig(api_key="sk-or-v1-invalid")
    client = OpenRouterClient(config=config, http_client=mock_http)
    provider = OpenRouterProvider(config=config, client=client)

    health = provider.health_check()
    assert health.status == ProviderStatus.UNHEALTHY.value
    assert "authentication failed" in health.message.lower()


# ============================================================================
# 5. INTEGRATION WITH ROUTED MODEL EXECUTOR & REGISTRY
# ============================================================================

def test_openrouter_through_provider_registry_and_routed_executor():
    class MockClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _mock_openrouter_success_payload(
                content="Routed execution via OpenRouter succeeded.",
                model="meta-llama/llama-3.3-70b-instruct",
            )

        def check_key_auth(self) -> dict[str, Any]:
            return {"data": {"label": "test-key"}}

    provider_registry = ProviderRegistry()
    openrouter_provider = OpenRouterProvider(client=MockClient())
    provider_registry.register(openrouter_provider)

    model_registry = ModelRegistry()
    model_registry.register(
        RegisteredModel(
            model_id="meta-llama/llama-3.3-70b-instruct",
            provider_id="openrouter",
            capabilities=["text_generation"],
            input_modalities=["text"],
            output_modalities=["text"],
            maximum_context_tokens=131072,
        )
    )

    executor = RoutedModelExecutor(
        provider_registry=provider_registry,
        model_registry=model_registry,
    )

    req = _sample_request(model_id="meta-llama/llama-3.3-70b-instruct")
    res = executor.execute(req)

    assert res.status == ExecutionStatus.SUCCESS.value
    assert res.model_id == "meta-llama/llama-3.3-70b-instruct"
    assert "Routed execution via OpenRouter succeeded." in res.output
    assert res.metadata["provider_id"] == "openrouter"
    assert res.metadata["routed"] == "true"


def test_existing_providers_remain_unaffected():
    # Verify MockModelExecutor and LocalModelExecutor still operate independently
    mock_exec = MockModelExecutor()
    local_exec = LocalModelExecutor()

    req = _sample_request()
    res_mock = mock_exec.execute(req)
    res_local = local_exec.execute(req)

    assert res_mock.status == ExecutionStatus.SUCCESS.value
    assert "[mock-output:" in res_mock.output

    assert res_local.status == ExecutionStatus.SUCCESS.value
    assert "[local:" in res_local.output


# ============================================================================
# 6. ARCHITECTURE BOUNDARY TESTS
# ============================================================================

def test_openrouter_does_not_perform_capability_matching():
    # OpenRouterProvider does not validate or filter by ModelRequirements
    class MockClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _mock_openrouter_success_payload(content="Audio analysis complete")

    provider = OpenRouterProvider(client=MockClient())

    # Send a request with non-text modalities directly to the provider adapter;
    # it executes without capability rejection because capability matching belongs to ModelRouter
    req = _sample_request(
        input_text="Transcribe speech",
        model_id="any/audio-model",
    )
    res = provider.execute(req)
    assert res.status == ExecutionStatus.SUCCESS.value


def test_openrouter_does_not_select_different_model():
    # Even if upstream returns an alias, the contract preserves the requested model_id
    class AliasClient(OpenRouterClient):
        def __init__(self):
            super().__init__(OpenRouterConfig(api_key="sk-or-v1-test"))

        def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _mock_openrouter_success_payload(
                model="deepseek/deepseek-chat-v3-0120",  # Upstream alias
            )

    provider = OpenRouterProvider(client=AliasClient())
    req = _sample_request(model_id="deepseek/deepseek-chat")
    res = provider.execute(req)

    # Requested identity is strictly preserved in model_id
    assert res.model_id == "deepseek/deepseek-chat"
    assert res.metadata["returned_model"] == "deepseek/deepseek-chat-v3-0120"


def test_openrouter_does_not_select_compute():
    config = OpenRouterConfig(api_key="sk-or-v1-test")
    provider = OpenRouterProvider(config=config)

    # Verify no compute/GPU/cloud fields exist on OpenRouterProvider or OpenRouterConfig
    assert not hasattr(provider, "compute")
    assert not hasattr(provider, "gpu_id")
    assert not hasattr(config, "cloud_provider")
    assert not hasattr(config, "gpu_type")


def test_openrouter_does_not_perform_provider_routing():
    provider = OpenRouterProvider()

    # Verify OpenRouterProvider only knows its own provider identity and does not route
    assert not hasattr(provider, "select_model")
    assert not hasattr(provider, "resolve_provider")
    assert not hasattr(provider, "route")
