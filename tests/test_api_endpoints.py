"""Tests for Prompt Engine REST API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_resolve_and_execute_incomplete(client):
    payload = {
        "promptIdentity": "system_bot_master_intent",
        "versionSelector": "ACTIVE",
        "rawInput": "Help me book an appointment tomorrow",
        "runtimeContext": {"master_id": "11111111-1111-1111-1111-111111111111"},
        "input": {},
    }
    response = await client.post("/v1/resolver/resolve-and-execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["promptIdentity"] == "system_bot_master_intent"
    assert data["isReady"] is False
    assert "start_time" in data["missingVariables"]
    assert "workplace_id" in data["missingVariables"]
    assert data["executionId"] is None


@pytest.mark.asyncio
async def test_resolve_and_execute_complete(client):
    booking_data = {
        "master_id": "11111111-1111-1111-1111-111111111111",
        "customer_id": "22222222-2222-2222-2222-222222222222",
        "workplace_id": "33333333-3333-3333-3333-333333333333",
        "start_time": "2026-10-01T10:00:00Z",
        "end_time": "2026-10-01T11:00:00Z",
        "items": [{"service_id": "44444444-4444-4444-4444-444444444444", "quantity": 1}],
    }
    payload = {
        "promptIdentity": "system_bot_master_intent",
        "versionSelector": "ACTIVE",
        "input": booking_data,
    }
    response = await client.post("/v1/resolver/resolve-and-execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["promptIdentity"] == "system_bot_master_intent"
    assert data["isReady"] is True
    assert data["missingVariables"] == []
    assert data["status"] == "SUCCESS"
    assert data["executionId"] is not None
    assert data["structuredOutput"] is not None
    assert data["structuredOutput"]["master_id"] == "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
async def test_resolve_endpoint_standalone(client):
    payload = {
        "promptIdentity": "system_bot_master_intent",
        "versionSelector": "ACTIVE",
        "input": {
            "master_id": "11111111-1111-1111-1111-111111111111",
        },
    }
    response = await client.post("/v1/resolver/resolve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["isReady"] is False
    assert "customer_id" in data["missingVariables"]


@pytest.mark.asyncio
async def test_unknown_prompt_returns_404(client):
    payload = {
        "promptIdentity": "unknown_nonexistent_prompt",
        "versionSelector": "ACTIVE",
    }
    response = await client.post("/v1/resolver/resolve-and-execute", json=payload)
    assert response.status_code == 404
