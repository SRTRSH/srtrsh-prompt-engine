"""Tests for variable binding and the mandatory Readiness Gate in the Resolver."""

import pytest

from srtrsh_prompt_engine.domain.models import VariableSource
from srtrsh_prompt_engine.domain.runtime import ResolutionRequest
from srtrsh_prompt_engine.resolver.service import ResolverService
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.registry import ProviderRegistry
from srtrsh_prompt_engine.storage.repository import InMemoryPromptRepository
from srtrsh_prompt_engine.seeds.system_bot import create_system_bot_master_intent_prompt


@pytest.mark.asyncio
async def test_readiness_gate_fails_when_variables_missing():
    repo = InMemoryPromptRepository()
    repo.save(create_system_bot_master_intent_prompt())

    provider_called = False

    def spy_handler(req):
        nonlocal provider_called
        provider_called = True
        return MockProvider().execute(req)

    registry = ProviderRegistry()
    registry.register("mock", MockProvider(custom_handler=spy_handler))

    service = ResolverService(repository=repo, provider_registry=registry)

    # Incomplete request: only master_id and customer_id provided
    request = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        input={
            "master_id": "11111111-1111-1111-1111-111111111111",
            "customer_id": "22222222-2222-2222-2222-222222222222",
        },
    )

    result = await service.resolve_and_execute(request)

    assert result.is_ready is False
    # Check that missing variables are identified
    assert "workplace_id" in result.missing_variables
    assert "start_time" in result.missing_variables
    assert "end_time" in result.missing_variables
    assert "items" in result.missing_variables

    # Verify bound variables
    bound_names = {b.name: b.value for b in result.bound_variables}
    assert bound_names["master_id"] == "11111111-1111-1111-1111-111111111111"
    assert bound_names["customer_id"] == "22222222-2222-2222-2222-222222222222"

    # MANDATORY GATE: Provider MUST NOT be called when isReady is False!
    assert provider_called is False
    assert result.execution_id is None
    assert result.structured_output is None


@pytest.mark.asyncio
async def test_binding_from_runtime_context_and_raw_input():
    repo = InMemoryPromptRepository()
    repo.save(create_system_bot_master_intent_prompt())
    service = ResolverService(repository=repo, provider_registry=ProviderRegistry())

    # master_id provided via runtime_context
    # customer_id extracted from raw_input
    # workplace_id, start_time, end_time, items in input
    request = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        runtime_context={"master_id": "11111111-1111-1111-1111-111111111111"},
        raw_input="customer_id: 22222222-2222-2222-2222-222222222222",
        input={
            "workplace_id": "33333333-3333-3333-3333-333333333333",
            "start_time": "2026-10-01T10:00:00Z",
            "end_time": "2026-10-01T11:00:00Z",
            "items": [{"service_id": "44444444-4444-4444-4444-444444444444", "quantity": 1}],
        },
    )

    result = await service.resolve_and_execute(request)
    assert result.is_ready is True
    assert result.missing_variables == []

    sources = {b.name: b.source for b in result.bound_variables}
    assert sources["master_id"] == VariableSource.CONTEXT_PAYLOAD
    assert sources["customer_id"] == VariableSource.RUNTIME_INPUT
    assert sources["workplace_id"] == VariableSource.RUNTIME_INPUT
