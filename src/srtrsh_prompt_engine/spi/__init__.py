"""Provider SPI module."""

from srtrsh_prompt_engine.spi.generic_llm import (
    GenericLLMProvider,
    normalize_http_error,
)
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.provider import (
    HydratedPrompt,
    LLMResponse,
    NormalizedError,
    NormalizedInteractionOutput,
    NormalizedOperationCall,
    ProviderCapabilities,
    ProviderErrorCategory,
    ProviderExecutionRequest,
    ProviderExecutionResponse,
    ProviderSPI,
)
from srtrsh_prompt_engine.spi.registry import ProviderRegistry

__all__ = [
    "ProviderSPI",
    "ProviderCapabilities",
    "HydratedPrompt",
    "ProviderExecutionRequest",
    "ProviderExecutionResponse",
    "LLMResponse",
    "NormalizedOperationCall",
    "NormalizedInteractionOutput",
    "NormalizedError",
    "ProviderErrorCategory",
    "MockProvider",
    "GenericLLMProvider",
    "ProviderRegistry",
    "normalize_http_error",
]
