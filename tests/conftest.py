"""Test configuration and shared fixtures for Prompt Engine."""

import pytest
from httpx import ASGITransport, AsyncClient

from srtrsh_prompt_engine.api.app import create_app
from srtrsh_prompt_engine.seeds.system_bot import create_system_bot_master_intent_prompt
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.registry import ProviderRegistry
from srtrsh_prompt_engine.storage.repository import InMemoryPromptRepository


@pytest.fixture
def repository():
    repo = InMemoryPromptRepository()
    seed_prompt = create_system_bot_master_intent_prompt()
    repo.save(seed_prompt)
    return repo


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def provider_registry(mock_provider):
    registry = ProviderRegistry()
    registry.register("mock", mock_provider)
    return registry


@pytest.fixture
def app(repository, provider_registry):
    return create_app(repository=repository, provider_registry=provider_registry)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
