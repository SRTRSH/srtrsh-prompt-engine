"""Deterministic Mock Provider for testing and local operation."""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from srtrsh_prompt_engine.domain.runtime import ExecutionUsageTelemetry
from srtrsh_prompt_engine.spi.provider import (
    NormalizedError,
    NormalizedInteractionOutput,
    NormalizedOperationCall,
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
        self._operation_calls: List[NormalizedOperationCall] = []
        self._interaction: Optional[NormalizedInteractionOutput] = None
        self._error: Optional[NormalizedError] = None
        self._call_history: List[ProviderExecutionRequest] = []

    def set_default_output(self, output: Optional[Dict[str, Any]]) -> None:
        self._default_output = output

    def set_operation_calls(self, calls: List[NormalizedOperationCall]) -> None:
        self._operation_calls = calls

    def set_interaction(self, interaction: Optional[NormalizedInteractionOutput]) -> None:
        self._interaction = interaction

    def set_error(self, error: Optional[NormalizedError]) -> None:
        self._error = error

    def set_custom_handler(
        self,
        handler: Optional[
            Callable[[ProviderExecutionRequest], ProviderExecutionResponse]
        ],
    ) -> None:
        self._custom_handler = handler

    def get_call_history(self) -> List[ProviderExecutionRequest]:
        return list(self._call_history)

    def clear_history(self) -> None:
        self._call_history.clear()

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
        self._call_history.append(request)

        if self._custom_handler:
            return self._custom_handler(request)

        if self._error:
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="FAILURE",
                error_message=self._error.message,
                normalized_error=self._error,
                usage=ExecutionUsageTelemetry(latency_ms=5.0),
            )

        if self._operation_calls:
            calls = list(self._operation_calls)
            # If calls are set, return them
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="SUCCESS",
                operation_calls=calls,
                raw_text="Requesting operation calls",
                usage=ExecutionUsageTelemetry(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    latency_ms=10.0,
                ),
                finish_reason="tool_calls",
            )

        if self._interaction:
            return ProviderExecutionResponse(
                execution_id=request.execution_id,
                status="SUCCESS",
                interaction=self._interaction,
                raw_text=self._interaction.text,
                usage=ExecutionUsageTelemetry(
                    input_tokens=80,
                    output_tokens=40,
                    total_tokens=120,
                    latency_ms=10.0,
                ),
                finish_reason="stop",
            )

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

        # Default fallback extraction
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
                if target_props is not None and k not in target_props:
                    continue
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
