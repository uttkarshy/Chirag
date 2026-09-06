"""OpenRouter Provider Adapter for Chirag (M3.6).

ARCHITECTURAL SEPARATION:
-------------------------
MODEL ("What"):
    The logical model identity and capability profile
    (e.g., 'deepseek/deepseek-chat', 'meta-llama/llama-3.3-70b-instruct').
    Defined in ModelRegistry and selected by ModelRouter based on ModelRequirements.

PROVIDER ("How"):
    The inference service / protocol adapter through which the model is accessed
    (e.g., 'openrouter').
    Registered in ProviderRegistry and invoked by RoutedModelExecutor.

COMPUTE ("Where"):
    The physical or virtual infrastructure where execution occurs
    (e.g., external SaaS API, private GPU instance, local workstation).
    Decided independently by future ComputeRouter / ComputeRegistry mechanisms.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..schema import (
    ExecutionStatus,
    Modality,
    ModelExecutionRequest,
    ModelResult,
    ProviderHealth,
    ProviderStatus,
)
from ..validation import (
    ProviderError,
    sanitize_error_message,
    validate_execution_request,
    validate_model_result,
)
from .base import ModelProvider


def _scrub_secrets(message: str, api_key: str | None = None) -> str:
    """Scrub authorization tokens and sensitive API keys from strings."""
    if not message:
        return message
    scrubbed = message
    if api_key and len(api_key) > 4:
        scrubbed = scrubbed.replace(api_key, "[REDACTED]")
    # Redact Bearer tokens
    scrubbed = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", scrubbed, flags=re.IGNORECASE)
    # Redact standard OpenRouter key prefixes
    scrubbed = re.sub(r"sk-or-v1-[a-zA-Z0-9]+", "[REDACTED]", scrubbed)
    return scrubbed


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter provider adapter."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = Field(
        default=None,
        description="OpenRouter API key. If None, dynamically resolved from OPENROUTER_API_KEY environment variable.",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter API.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Network timeout in seconds for chat completions.",
    )
    health_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Network timeout in seconds for lightweight health checks.",
    )
    site_url: str | None = Field(
        default=None,
        description="Optional HTTP-Referer header for OpenRouter analytics/rankings.",
    )
    app_name: str | None = Field(
        default=None,
        description="Optional X-Title header for OpenRouter analytics/rankings.",
    )


class OpenRouterClient:
    """Provider-local HTTP client for OpenRouter API communication."""

    def __init__(
        self,
        config: OpenRouterConfig,
        http_client: httpx.Client | Any = None,
    ) -> None:
        self.config = config
        self._http_client = http_client

    def get_api_key(self) -> str:
        """Resolve API key from config or runtime environment. Never hard-coded."""
        key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key or not key.strip():
            raise ProviderError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY environment variable or configure OpenRouterConfig.api_key."
            )
        return key.strip()

    def _get_headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name
        return headers

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch OpenAI-compatible chat completion POST request."""
        api_key = self.get_api_key()
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = self._get_headers(api_key)

        try:
            if self._http_client is not None:
                resp = self._http_client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    resp = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"OpenRouter request timed out after {self.config.timeout_seconds}s."
            ) from exc
        except httpx.ConnectError as exc:
            sanitized = _scrub_secrets(sanitize_error_message(exc), api_key)
            raise ProviderError(f"OpenRouter connection failed: {sanitized}") from exc
        except httpx.HTTPError as exc:
            sanitized = _scrub_secrets(sanitize_error_message(exc), api_key)
            raise ProviderError(f"OpenRouter network error: {sanitized}") from exc
        except Exception as exc:
            sanitized = _scrub_secrets(sanitize_error_message(exc), api_key)
            raise ProviderError(f"OpenRouter client error: {sanitized}") from exc

        # Handle non-200 HTTP status
        if resp.status_code != 200:
            error_detail = ""
            try:
                err_data = resp.json()
                if isinstance(err_data, dict) and "error" in err_data:
                    err_obj = err_data["error"]
                    if isinstance(err_obj, dict):
                        error_detail = err_obj.get("message") or str(err_obj)
                    else:
                        error_detail = str(err_obj)
            except Exception:
                error_detail = resp.text[:200] if hasattr(resp, "text") else ""

            sanitized_detail = _scrub_secrets(error_detail, api_key)
            if resp.status_code == 401:
                raise ProviderError(
                    "OpenRouter authentication failed: Invalid or unauthorized API key."
                )
            if sanitized_detail:
                raise ProviderError(
                    f"OpenRouter API error ({resp.status_code}): {sanitized_detail}"
                )
            raise ProviderError(
                f"OpenRouter returned non-success HTTP status ({resp.status_code})."
            )

        # Parse and return JSON
        try:
            return resp.json()
        except Exception as exc:
            raise ProviderError("OpenRouter returned malformed JSON response.") from exc

    def check_key_auth(self) -> dict[str, Any]:
        """Perform lightweight check against OpenRouter key verification endpoint without generation."""
        api_key = self.get_api_key()
        url = f"{self.config.base_url.rstrip('/')}/auth/key"
        headers = self._get_headers(api_key)

        try:
            if self._http_client is not None:
                resp = self._http_client.get(
                    url,
                    headers=headers,
                    timeout=self.config.health_timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.config.health_timeout_seconds) as client:
                    resp = client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"OpenRouter health check timed out after {self.config.health_timeout_seconds}s."
            ) from exc
        except Exception as exc:
            sanitized = _scrub_secrets(sanitize_error_message(exc), api_key)
            raise ProviderError(
                f"OpenRouter health check connection failed: {sanitized}"
            ) from exc

        if resp.status_code == 401:
            raise ProviderError("OpenRouter authentication failed: Invalid API key.")
        if resp.status_code != 200:
            raise ProviderError(
                f"OpenRouter health check returned HTTP {resp.status_code}."
            )

        try:
            return resp.json()
        except Exception as exc:
            raise ProviderError(
                "OpenRouter health check returned malformed JSON."
            ) from exc


class OpenRouterProvider(ModelProvider):
    """Production-quality ModelProvider adapter for OpenRouter."""

    def __init__(
        self,
        provider_id: str = "openrouter",
        config: OpenRouterConfig | None = None,
        client: OpenRouterClient | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.provider_type = "openrouter"
        self.config = config or (client.config if client is not None else OpenRouterConfig())
        self.client = client or OpenRouterClient(self.config)

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        """Execute the request via OpenRouter and normalize into a ModelResult."""
        if not isinstance(request, ModelExecutionRequest):
            raise ValueError(
                f"Invalid request type: expected ModelExecutionRequest, got {type(request).__name__}"
            )
        validate_execution_request(request)

        primary_out = (
            request.output_modalities[0]
            if request.output_modalities
            else Modality.TEXT.value
        )

        # Map ModelExecutionRequest to OpenAI-compatible OpenRouter payload
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                {"role": "user", "content": request.input}
            ],
        }
        if request.structured_output:
            payload["response_format"] = {"type": "json_object"}
        if request.streaming:
            payload["stream"] = True

        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")

        # Execute remote call via client abstraction
        try:
            raw_resp = self.client.create_chat_completion(payload)
        except httpx.TimeoutException as exc:
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={
                    "prompt_tokens": len(request.input.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(request.input.split()),
                },
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error=f"OpenRouter request timed out after {self.config.timeout_seconds}s.",
            )
            return validate_model_result(failed_res)
        except httpx.ConnectError as exc:
            sanitized = _scrub_secrets(sanitize_error_message(exc), api_key)
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={
                    "prompt_tokens": len(request.input.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(request.input.split()),
                },
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error=f"OpenRouter connection failed: {sanitized}",
            )
            return validate_model_result(failed_res)
        except Exception as exc:
            err_msg = _scrub_secrets(sanitize_error_message(exc), api_key)
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={
                    "prompt_tokens": len(request.input.split()),
                    "completion_tokens": 0,
                    "total_tokens": len(request.input.split()),
                },
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error=err_msg,
            )
            return validate_model_result(failed_res)

        # Validate response format
        if not isinstance(raw_resp, dict):
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error="OpenRouter response is not a valid dictionary.",
            )
            return validate_model_result(failed_res)

        choices = raw_resp.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error="OpenRouter response missing choices array.",
            )
            return validate_model_result(failed_res)

        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error="OpenRouter choice missing message object.",
            )
            return validate_model_result(failed_res)

        message_obj = first_choice.get("message")
        if not isinstance(message_obj, dict):
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error="OpenRouter message is not a dictionary.",
            )
            return validate_model_result(failed_res)

        output_content = message_obj.get("content")
        if output_content is None or not str(output_content).strip():
            failed_res = ModelResult(
                model_id=request.model_id,
                status=ExecutionStatus.FAILED.value,
                output=None,
                output_modality=primary_out,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                },
                error="OpenRouter returned empty or null content.",
            )
            return validate_model_result(failed_res)

        # Normalize usage metrics
        usage_raw = raw_resp.get("usage") or {}
        prompt_tokens = int(usage_raw.get("prompt_tokens", len(request.input.split())))
        completion_tokens = int(usage_raw.get("completion_tokens", len(str(output_content).split())))
        total_tokens = int(usage_raw.get("total_tokens", 0)) or (prompt_tokens + completion_tokens)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        # Normalize metadata; preserve returned_model if different from requested model_id
        metadata: dict[str, str] = {
            **request.metadata,
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
        }
        returned_model = raw_resp.get("model")
        if returned_model:
            metadata["returned_model"] = str(returned_model)
        if "id" in raw_resp:
            metadata["openrouter_id"] = str(raw_resp["id"])
        if "provider" in raw_resp:
            metadata["upstream_provider"] = str(raw_resp["provider"])
        if request.structured_output:
            metadata["structured_output"] = "true"
        if request.streaming:
            metadata["streaming"] = "true"

        success_res = ModelResult(
            model_id=request.model_id,
            status=ExecutionStatus.SUCCESS.value,
            output=str(output_content),
            output_modality=primary_out,
            usage=usage,
            metadata=metadata,
            error=None,
        )
        return validate_model_result(success_res)

    def health_check(self) -> ProviderHealth:
        """Perform lightweight health check against OpenRouter key endpoint without generation."""
        try:
            self.client.get_api_key()
        except Exception:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderStatus.UNHEALTHY.value,
                latency_ms=None,
                message="OpenRouter API key is not configured.",
                metadata={"provider_type": self.provider_type},
            )

        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
        start_time = time.perf_counter()
        try:
            self.client.check_key_auth()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            status = (
                ProviderStatus.HEALTHY.value
                if latency_ms < 3000.0
                else ProviderStatus.DEGRADED.value
            )
            return ProviderHealth(
                provider_id=self.provider_id,
                status=status,
                latency_ms=round(latency_ms, 2),
                message="OpenRouter operational",
                metadata={"provider_type": self.provider_type},
            )
        except Exception as exc:
            err_msg = _scrub_secrets(sanitize_error_message(exc), api_key)
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderStatus.UNHEALTHY.value,
                latency_ms=None,
                message=f"OpenRouter health check failed: {err_msg}",
                metadata={"provider_type": self.provider_type},
            )
