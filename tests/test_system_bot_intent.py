"""Integration tests for Prompt Identity 'system_bot_master_intent' and operation 'DirectBookingRequest'."""

import pytest

from srtrsh_prompt_engine.domain.runtime import ExecutionStatus, ResolutionRequest
from srtrsh_prompt_engine.resolver.service import ResolverService
from srtrsh_prompt_engine.seeds.system_bot import create_system_bot_master_intent_prompt
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.registry import ProviderRegistry
from srtrsh_prompt_engine.storage.repository import InMemoryPromptRepository


@pytest.fixture
def system_bot_service():
    repo = InMemoryPromptRepository()
    repo.save(create_system_bot_master_intent_prompt())
    provider = MockProvider()
    registry = ProviderRegistry()
    registry.register("mock", provider)
    return ResolverService(repository=repo, provider_registry=registry), provider


@pytest.mark.asyncio
async def test_system_bot_incomplete_direct_booking_request(system_bot_service):
    service, _ = system_bot_service

    # Master says: "Запиши клиента на завтра"
    # Channel Adapter resolves master_id from Channel Identity and supplies raw text
    request = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        raw_input="Запиши клиента на завтра",
        runtime_context={"master_id": "11111111-1111-1111-1111-111111111111"},
        input={},
    )

    result = await service.resolve_and_execute(request)

    # Incomplete result: cannot execute yet
    assert result.is_ready is False
    assert result.prompt_identity == "system_bot_master_intent"
    assert result.resolved_version == 1
    assert result.version_hash is not None

    # master_id was resolved from context, but customer_id, workplace_id, start_time, end_time, items remain missing
    assert "customer_id" in result.missing_variables
    assert "workplace_id" in result.missing_variables
    assert "start_time" in result.missing_variables
    assert "end_time" in result.missing_variables
    assert "items" in result.missing_variables
    assert "master_id" not in result.missing_variables


@pytest.mark.asyncio
async def test_system_bot_complete_direct_booking_request(system_bot_service):
    service, provider = system_bot_service

    complete_payload = {
        "master_id": "99999999-9999-9999-9999-999999999999",
        "customer_id": "88888888-8888-8888-8888-888888888888",
        "workplace_id": "77777777-7777-7777-7777-777777777777",
        "start_time": "2026-10-15T09:00:00Z",
        "end_time": "2026-10-15T10:00:00Z",
        "items": [
            {
                "service_id": "66666666-6666-6666-6666-666666666666",
                "quantity": 1,
            }
        ],
    }
    provider.set_default_output(complete_payload)

    request = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        runtime_context={"master_id": "99999999-9999-9999-9999-999999999999"},
        input={
            "customer_id": "88888888-8888-8888-8888-888888888888",
            "workplace_id": "77777777-7777-7777-7777-777777777777",
            "start_time": "2026-10-15T09:00:00Z",
            "end_time": "2026-10-15T10:00:00Z",
            "items": [
                {
                    "service_id": "66666666-6666-6666-6666-666666666666",
                    "quantity": 1,
                }
            ],
        },
    )

    result = await service.resolve_and_execute(request)

    # Complete structured result
    assert result.is_ready is True
    assert result.missing_variables == []
    assert result.status == ExecutionStatus.SUCCESS
    assert result.structured_output == complete_payload
