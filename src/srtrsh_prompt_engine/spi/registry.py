"""Provider Registry for managing ProviderSPI implementations."""

from typing import Dict, Optional

from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.provider import ProviderSPI


class ProviderRegistry:
    """Registry maintaining active provider implementations."""

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderSPI] = {}
        # Register default mock provider
        self.register("mock", MockProvider())

    def register(self, provider_id: str, provider: ProviderSPI) -> None:
        self._providers[provider_id.lower()] = provider

    def get(self, provider_id: Optional[str]) -> Optional[ProviderSPI]:
        if not provider_id:
            return self._providers.get("mock")
        return self._providers.get(provider_id.lower())

    def get_or_default(self, provider_id: Optional[str]) -> ProviderSPI:
        p = self.get(provider_id)
        if p is None:
            # Fall back to mock if not found
            return self._providers.get("mock", MockProvider())
        return p
