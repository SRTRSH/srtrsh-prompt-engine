"""Provider Service Provider Interface (SPI).

Normative reference:
WikiDocs/14-prompt-engine/03-api-contracts/03-provider-spi.md
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from srtrsh_prompt_engine.domain.runtime import ExecutionUsageTelemetry


class ProviderCapabilities(BaseModel):
    supports_structured_output: bool = True
    supports_system_instruction: bool = True
    max_context_tokens: int = 128000


class HydratedPrompt(BaseModel):
    system_prompt: str
    user_prompt: str


class ProviderExecutionRequest(BaseModel):
    execution_id: str
    prompt_version_hash: str
    provider: str
    model: str
    prompt: HydratedPrompt
    target_schema: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProviderExecutionResponse(BaseModel):
    execution_id: str
    status: str = "SUCCESS"  # SUCCESS or FAILURE
    output: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    usage: ExecutionUsageTelemetry = Field(default_factory=ExecutionUsageTelemetry)
    finish_reason: Optional[str] = "stop"
    error_message: Optional[str] = None
    provider_metadata: Dict[str, Any] = Field(default_factory=dict)


class ProviderSPI(ABC):
    """Abstract technical abstraction boundary between Prompt Engine and LLM providers."""

    @abstractmethod
    async def execute(
        self, request: ProviderExecutionRequest
    ) -> ProviderExecutionResponse:
        """Invoke external provider and return normalized response."""
        ...

    @abstractmethod
    def get_capabilities(
        self, model: Optional[str] = None
    ) -> ProviderCapabilities:
        """Return capabilities supported by this provider/model."""
        ...
