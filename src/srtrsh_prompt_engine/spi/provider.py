"""Provider Service Provider Interface (SPI).

Normative references:
WikiDocs/14-prompt-engine/13-provider-spi/01-provider-contract.md
WikiDocs/14-prompt-engine/13-provider-spi/02-request-normalization.md
WikiDocs/14-prompt-engine/13-provider-spi/03-response-normalization.md
WikiDocs/14-prompt-engine/13-provider-spi/04-error-normalization.md
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from srtrsh_prompt_engine.domain.runtime import ExecutionUsageTelemetry


class ProviderErrorCategory(str, Enum):
    """Normalized Prompt Engine provider error categories (C-15)."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_FAILURE = "authentication_failure"
    CONTENT_POLICY_VIOLATION = "content_policy_violation"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    TECHNICAL_ERROR = "technical_error"


class NormalizedError(BaseModel):
    """Normalized provider error model (Master section 57, C-15)."""

    category: ProviderErrorCategory
    message: str
    retryable: bool
    provider_id: Optional[str] = None
    model_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class NormalizedOperationCall(BaseModel):
    """Provider-independent representation of an AI operation request (Master section 56, C-28).

    Guarantees provider-native tool/function-calling formats do not leak into core.
    """

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="callId")
    operation_id: str = Field(..., alias="operationId")
    arguments: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class NormalizedInteractionOutput(BaseModel):
    """Structured interactive response options."""

    text: str
    options: List[Dict[str, Any]] = Field(default_factory=list)


class ProviderCapabilities(BaseModel):
    supports_structured_output: bool = True
    supports_system_instruction: bool = True
    supports_tool_calling: bool = True
    max_context_tokens: int = 128000


class HydratedPrompt(BaseModel):
    system_prompt: str
    user_prompt: str


class ProviderExecutionRequest(BaseModel):
    """Provider-independent input for one LLM invocation (Master section 56, C-14)."""

    execution_id: str
    prompt_version_hash: str
    provider: str
    model: str
    prompt: HydratedPrompt
    target_schema: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    available_operations: List[str] = Field(default_factory=list)
    operation_results: List[Dict[str, Any]] = Field(default_factory=list)


class ProviderExecutionResponse(BaseModel):
    """Normalized LLM response model (Master section 56, 03-response-normalization.md, C-28)."""

    execution_id: str
    status: str = "SUCCESS"  # SUCCESS or FAILURE
    content: Optional[str] = None
    raw_text: Optional[str] = None
    operation_calls: List[NormalizedOperationCall] = Field(
        default_factory=list, alias="operationCalls"
    )
    interaction: Optional[NormalizedInteractionOutput] = None
    output: Optional[Dict[str, Any]] = None  # Structured output if applicable
    usage: ExecutionUsageTelemetry = Field(default_factory=ExecutionUsageTelemetry)
    finish_reason: Optional[str] = "stop"
    error_message: Optional[str] = None
    normalized_error: Optional[NormalizedError] = None
    provider_metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


# Backward compatibility alias
LLMResponse = ProviderExecutionResponse


class ProviderSPI(ABC):
    """Abstract technical abstraction boundary between Prompt Engine and LLM providers (C-14)."""

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
