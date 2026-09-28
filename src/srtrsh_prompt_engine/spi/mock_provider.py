"""Deterministic Mock Provider for testing and local operation."""

import json
import re
from typing import Any, Callable, Dict, Optional

from srtrsh_prompt_engine.domain.runtime import ExecutionUsageTelemetry
from srtrsh_prompt_engine.spi.provider import (
    ProviderCapabilities,
    ProviderExecutionRequest,
    ProviderExecutionResponse,
    ProviderSPI,
)


class MockProvider(ProviderSPI):
    """Deterministic mock provider satisfying ProviderSPI without external network calls."""

    def __init__(
        self,
        default_output: Optional[Dict[str, Any]] = None,
        custom_handler: Optional[
            Callable[[ProviderExecutionRequest], ProviderExecutionResponse]
        ] = None,
    ):
        self._default_output = default_output
        self._custom_handler = custom_handler

    def set_default_output(self, output: Optional[Dict[str, Any]]) -> None:
        self._default_output = output

    def set_custom_handler(
        self,
        handler: Optional[
            Callable[[ProviderExecutionRequest], ProviderExecutionResponse]
        ],
    ) -> None:
        self._custom_handler = handler

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
        if self._custom_handler:
            return self._custom_handler(request)

        if self._default_output is not None:
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="SUCCESS",
                output=self._default_output,
                raw_text=json.dumps(self._default_output),
                usage=ExecutionUsageTelemetry(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    latency_ms=10.0,
                ),
                finish_reason="stop",
            )

        # Default intelligent extraction: attempt to parse variables from user_prompt
        # Format usually: "Key: Value" lines
        parsed_fields: Dict[str, Any] = {}
        target_props = (
            request.target_schema.get("properties")
            if (request.target_schema and isinstance(request.target_schema, dict))
            else None
        )

        for line in request.prompt.user_prompt.splitlines():
            match = re.match(r"^([a-zA-Z0-9_]+):\s*(.*)$", line.strip())
            if match:
                k, v = match.group(1), match.group(2)
                # If target schema defines properties, only include those properties
                if target_props is not None and k not in target_props:
                    continue
                # Attempt to parse json value
                try:
                    parsed_fields[k] = json.loads(v)
                except Exception:
                    parsed_fields[k] = v

        return ProviderExecutionResponse(
            execution_id=request.execution_id,
            status="SUCCESS",
            output=parsed_fields,
            raw_text=json.dumps(parsed_fields),
            usage=ExecutionUsageTelemetry(
                input_tokens=50,
                output_tokens=30,
                total_tokens=80,
                latency_ms=5.0,
            ),
            finish_reason="stop",
        )
