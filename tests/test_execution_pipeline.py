"""Tests for Execution Pipeline, Provider SPI dispatch, Schema Validation, and Runtime Overrides."""

import pytest

from srtrsh_prompt_engine.domain.runtime import ExecutionStatus, ResolutionRequest
from srtrsh_prompt_engine.resolver.service import ResolverService
from srtrsh_prompt_engine.seeds.system_bot import create_system_bot_master_intent_prompt
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.provider import ProviderExecutionResponse
from srtrsh_prompt_engine.spi.registry import ProviderRegistry
from srtrsh_prompt_engine.storage.repository import InMemoryPromptRepository


@pytest.fixture
def test_setup():
    repo = InMemoryPromptRepository()
    repo.save(create_system_bot_master_intent_prompt())
    provider = MockProvider()
    registry = ProviderRegistry()
    registry.register("mock", provider)
    service = ResolverService(repository=repo, provider_registry=registry)
    return service, provider


@pytest.mark.asyncio
async def test_execution_succeeds_with_valid_schema(test_setup):
    service, provider = test_setup

    valid_booking_payload = {
        "master_id": "11111111-1111-1111-1111-111111111111",
        "customer_id": "22222222-2222-2222-2222-222222222222",
        "workplace_id": "33333333-3333-3333-3333-333333333333",
        "start_time": "2026-10-01T10:00:00Z",
        "end_time": "2026-10-01T11:00:00Z",
        "items": [
            {
                "service_id": "44444444-4444-4444-4444-444444444444",
                "quantity": 1,
            }
        ],
    }
    provider.set_default_output(valid_booking_payload)

    request = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        input=valid_booking_payload,
    )

    result = await service.resolve_and_execute(request)

    assert result.is_ready is True
    assert result.status == ExecutionStatus.SUCCESS
    assert result.structured_output == valid_booking_payload
    assert result.error_outcome is None
    assert result.usage_telemetry is not None
    assert result.usage_telemetry.latency_ms >= 0


@pytest.mark.asyncio
async def test_execution_fails_on_schema_violation(test_setup):
    service, provider = test_setup

    # Items empty array violates minItems: 1
    invalid_payload = {
        "master_id": "11111111-1111-1111-1111-111111111111",
        "customer_id": "22222222-2222-2222-2222-222222222222",
        "workplace_id": "33333333-3333-3333-3333-333333333333",
        "start_time": "2026-10-01T10:00:00Z",
        "end_time": "2026-10-01T11:00:00Z",
        "items": [],  # Violates minItems 1
    }
    provider.set_default_output(invalid_payload)

    request = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        input={
            "master_id": "11111111-1111-1111-1111-111111111111",
            "customer_id": "22222222-2222-2222-2222-222222222222",
            "workplace_id": "33333333-3333-3333-3333-333333333333",
            "start_time": "2026-10-01T10:00:00Z",
            "end_time": "2026-10-01T11:00:00Z",
            "items": [{"service_id": "44444444-4444-4444-4444-444444444444", "quantity": 1}],
        },
    )

    result = await service.resolve_and_execute(request)

    assert result.is_ready is True
    assert result.status == ExecutionStatus.TECHNICAL_FAILURE
    assert result.error_outcome is not None
    assert result.error_outcome.code == "SCHEMA_VALIDATION_ERROR"
    assert "non-empty" in result.error_outcome.message or "items" in result.error_outcome.message


@pytest.mark.asyncio
async def test_runtime_override_policy_enforcement(test_setup):
    service, provider = test_setup

    valid_input = {
        "master_id": "11111111-1111-1111-1111-111111111111",
        "customer_id": "22222222-2222-2222-2222-222222222222",
        "workplace_id": "33333333-3333-3333-3333-333333333333",
        "start_time": "2026-10-01T10:00:00Z",
        "end_time": "2026-10-01T11:00:00Z",
        "items": [{"service_id": "44444444-4444-4444-4444-444444444444", "quantity": 1}],
    }
    provider.set_default_output(valid_input)

    # 1. Permitted override: temperature is in allowed_parameters
    req_allowed = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        input=valid_input,
        runtime_overrides={"temperature": 0.5},
    )
    res_allowed = await service.resolve_and_execute(req_allowed)
    assert res_allowed.status == ExecutionStatus.SUCCESS

    # 2. Prohibited override: schemaContract or prohibited parameter
    req_forbidden = ResolutionRequest(
        prompt_identity="system_bot_master_intent",
        input=valid_input,
        runtime_overrides={"schemaContract": {}},
    )
    with pytest.raises(ValueError, match="not permitted by PromptVersion policy"):
        await service.resolve_and_execute(req_forbidden)
