"""Generic LLM provider adapter for standard OpenAI-compatible API endpoints.

Normative references:
WikiDocs/14-prompt-engine/13-provider-spi/01-provider-contract.md
WikiDocs/14-prompt-engine/13-provider-spi/02-request-normalization.md
WikiDocs/14-prompt-engine/13-provider-spi/03-response-normalization.md
WikiDocs/14-prompt-engine/13-provider-spi/04-error-normalization.md
"""

import json
import time
from typing import Any, Dict, List, Optional
import uuid

import httpx

from srtrsh_prompt_engine.domain.runtime import ExecutionUsageTelemetry
from srtrsh_prompt_engine.spi.provider import (
    NormalizedError,
    NormalizedOperationCall,
    ProviderCapabilities,
    ProviderErrorCategory,
    ProviderExecutionRequest,
    ProviderExecutionResponse,
    ProviderSPI,
)


def normalize_http_error(
    status_code: int, response_text: str, provider_id: str, model_name: str
) -> NormalizedError:
    """Normalize provider HTTP error codes into Prompt Engine error categories (C-15)."""
    text_lower = response_text.lower()
    if status_code == 408:
        return NormalizedError(
            category=ProviderErrorCategory.TIMEOUT,
            message="Provider request timed out",
            retryable=True,
            provider_id=provider_id,
            model_name=model_name,
            details={"raw_status": status_code, "raw_response": response_text},
        )
    elif status_code == 429:
        if "quota" in text_lower or "insufficient_quota" in text_lower:
            return NormalizedError(
                category=ProviderErrorCategory.QUOTA_EXHAUSTED,
                message="Provider quota exhausted",
                retryable=False,
                provider_id=provider_id,
                model_name=model_name,
                details={"raw_status": status_code, "raw_response": response_text},
            )
        return NormalizedError(
            category=ProviderErrorCategory.RATE_LIMIT,
            message="Provider rate limit exceeded",
            retryable=True,
            provider_id=provider_id,
            model_name=model_name,
            details={"raw_status": status_code, "raw_response": response_text},
        )
    elif status_code in (401, 403):
        return NormalizedError(
            category=ProviderErrorCategory.AUTHENTICATION_FAILURE,
            message="Authentication or authorization failed with provider",
            retryable=False,
            provider_id=provider_id,
            model_name=model_name,
            details={"raw_status": status_code, "raw_response": response_text},
        )
    elif status_code == 400:
        if "context_length_exceeded" in text_lower or "maximum context length" in text_lower:
            return NormalizedError(
                category=ProviderErrorCategory.CONTEXT_LENGTH_EXCEEDED,
                message="Context length exceeded provider limit",
                retryable=False,
                provider_id=provider_id,
                model_name=model_name,
                details={"raw_status": status_code, "raw_response": response_text},
            )
        if "content policy" in text_lower or "safety" in text_lower:
            return NormalizedError(
                category=ProviderErrorCategory.CONTENT_POLICY_VIOLATION,
                message="Content rejected by provider safety policy",
                retryable=False,
                provider_id=provider_id,
                model_name=model_name,
                details={"raw_status": status_code, "raw_response": response_text},
            )
        return NormalizedError(
            category=ProviderErrorCategory.INVALID_REQUEST,
            message=f"Invalid request sent to provider: {response_text}",
            retryable=False,
            provider_id=provider_id,
            model_name=model_name,
            details={"raw_status": status_code, "raw_response": response_text},
        )
    elif status_code in (500, 502, 503, 504):
        return NormalizedError(
            category=ProviderErrorCategory.PROVIDER_UNAVAILABLE,
            message=f"Provider temporary server error (HTTP {status_code})",
            retryable=True,
            provider_id=provider_id,
            model_name=model_name,
            details={"raw_status": status_code, "raw_response": response_text},
        )
    else:
        return NormalizedError(
            category=ProviderErrorCategory.TECHNICAL_ERROR,
            message=f"Unexpected provider status HTTP {status_code}: {response_text}",
            retryable=False,
            provider_id=provider_id,
            model_name=model_name,
            details={"raw_status": status_code, "raw_response": response_text},
        )


class GenericLLMProvider(ProviderSPI):
    """Adapter invoking OpenAI-compatible chat completion endpoints.

    Ensures provider SDKs remain behind the Provider SPI (C-14) and normalizes responses (C-28).
    """

    def __init__(
        self,
        api_base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model

    def get_capabilities(
        self, model: Optional[str] = None
    ) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_structured_output=True,
            supports_system_instruction=True,
            supports_tool_calling=True,
            max_context_tokens=128000,
        )

    async def execute(
        self, request: ProviderExecutionRequest
    ) -> ProviderExecutionResponse:
        model_name = request.model or self.default_model
        provider_id = request.provider or "openai"

        if not self.api_key:
            err = NormalizedError(
                category=ProviderErrorCategory.AUTHENTICATION_FAILURE,
                message="API key not configured for GenericLLMProvider",
                retryable=False,
                provider_id=provider_id,
                model_name=model_name,
            )
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="FAILURE",
                error_message=err.message,
                normalized_error=err,
            )

        messages = [
            {"role": "system", "content": request.prompt.system_prompt},
            {"role": "user", "content": request.prompt.user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }

        if request.parameters.get("temperature") is not None:
            payload["temperature"] = request.parameters["temperature"]
        if request.parameters.get("max_output_tokens") is not None:
            payload["max_tokens"] = request.parameters["max_output_tokens"]

        if request.target_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": request.target_schema,
                },
            }

        start_time = time.monotonic()
        timeout_sec = (request.parameters.get("timeout_duration") or 30000) / 1000.0

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                res = await client.post(
                    f"{self.api_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                latency_ms = (time.monotonic() - start_time) * 1000.0

                if res.status_code != 200:
                    normalized_err = normalize_http_error(
                        res.status_code, res.text, provider_id, model_name
                    )
                    return ProviderExecutionResponse(
                        execution_id=request.execution_id,
                        status="FAILURE",
                        error_message=normalized_err.message,
                        normalized_error=normalized_err,
                        usage=ExecutionUsageTelemetry(latency_ms=latency_ms),
                    )

                data = res.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content")
                usage_info = data.get("usage", {})

                # Normalize provider-native tool calls to NormalizedOperationCall (C-28)
                normalized_op_calls: List[NormalizedOperationCall] = []
                native_tool_calls = msg.get("tool_calls") or []
                for tc in native_tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        parsed_args = {}
                    normalized_op_calls.append(
                        NormalizedOperationCall(
                            call_id=tc.get("id") or str(uuid.uuid4()),
                            operation_id=fn_name,
                            arguments=parsed_args,
                        )
                    )

                parsed_json = None
                if content:
                    try:
                        parsed_json = json.loads(content)
                    except Exception:
                        parsed_json = None

                return ProviderExecutionResponse(
                    execution_id=request.execution_id,
                    status="SUCCESS",
                    content=content,
                    output=parsed_json,
                    raw_text=content,
                    operation_calls=normalized_op_calls,
                    usage=ExecutionUsageTelemetry(
                        input_tokens=usage_info.get("prompt_tokens", 0),
                        output_tokens=usage_info.get("completion_tokens", 0),
                        total_tokens=usage_info.get("total_tokens", 0),
                        latency_ms=latency_ms,
                    ),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
        except httpx.TimeoutException as exc:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            norm_err = NormalizedError(
                category=ProviderErrorCategory.TIMEOUT,
                message=f"Request to provider timed out: {exc}",
                retryable=True,
                provider_id=provider_id,
                model_name=model_name,
            )
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="FAILURE",
                error_message=norm_err.message,
                normalized_error=norm_err,
                usage=ExecutionUsageTelemetry(latency_ms=latency_ms),
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            norm_err = NormalizedError(
                category=ProviderErrorCategory.TECHNICAL_ERROR,
                message=str(exc),
                retryable=False,
                provider_id=provider_id,
                model_name=model_name,
            )
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="FAILURE",
                error_message=str(exc),
                normalized_error=norm_err,
                usage=ExecutionUsageTelemetry(latency_ms=latency_ms),
            )
