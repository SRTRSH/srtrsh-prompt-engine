"""Generic LLM provider adapter for standard OpenAI-compatible API endpoints."""

import json
import time
from typing import Any, Dict, Optional

import httpx

from srtrsh_prompt_engine.domain.runtime import ExecutionUsageTelemetry
from srtrsh_prompt_engine.spi.provider import (
    ProviderCapabilities,
    ProviderExecutionRequest,
    ProviderExecutionResponse,
    ProviderSPI,
)


class GenericLLMProvider(ProviderSPI):
    """Adapter invoking OpenAI-compatible chat completion endpoints."""

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
            max_context_tokens=128000,
        )

    async def execute(
        self, request: ProviderExecutionRequest
    ) -> ProviderExecutionResponse:
        if not self.api_key:
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="FAILURE",
                error_message="API key not configured for GenericLLMProvider",
            )

        messages = [
            {"role": "system", "content": request.prompt.system_prompt},
            {"role": "user", "content": request.prompt.user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": request.model or self.default_model,
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
                    return ProviderExecutionResponse(
                        execution_id=request.execution_id,
                        status="FAILURE",
                        error_message=f"HTTP {res.status_code}: {res.text}",
                        usage=ExecutionUsageTelemetry(latency_ms=latency_ms),
                    )

                data = res.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                parsed_json = json.loads(content) if content else {}
                usage_info = data.get("usage", {})

                return ProviderExecutionResponse(
                    execution_id=request.execution_id,
                    status="SUCCESS",
                    output=parsed_json,
                    raw_text=content,
                    usage=ExecutionUsageTelemetry(
                        input_tokens=usage_info.get("prompt_tokens", 0),
                        output_tokens=usage_info.get("completion_tokens", 0),
                        total_tokens=usage_info.get("total_tokens", 0),
                        latency_ms=latency_ms,
                    ),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="FAILURE",
                error_message=str(exc),
                usage=ExecutionUsageTelemetry(latency_ms=latency_ms),
            )
