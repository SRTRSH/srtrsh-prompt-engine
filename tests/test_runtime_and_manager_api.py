"""Tests for Runtime API, Manager API, and Localization API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_runtime_execute_endpoint(client):
    payload = {
        "conversationId": "conv_api_exec_1",
        "message": "Привет, хочу записаться",
        "caseKey": "resolve_user_request",
    }
    response = await client.post("/v1/runtime/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conversationId"] == "conv_api_exec_1"
    assert "executionId" in data
    assert data["status"] in ("COMPLETED", "NEEDS_USER_INPUT", "INTERACTIVE_RESPONSE", "HUMAN_FALLBACK")
    assert data["sessionVersion"] >= 1


@pytest.mark.asyncio
async def test_runtime_interaction_endpoint(client):
    payload = {
        "conversationId": "conv_api_interact_1",
        "interactionResult": {"confirmation": True},
        "caseKey": "resolve_user_request",
    }
    response = await client.post("/v1/runtime/interaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conversationId"] == "conv_api_interact_1"
    assert data["structuredState"].get("confirmation") is True


@pytest.mark.asyncio
async def test_manager_api_prompt_cases_lifecycle(client):
    # 1. Create a version for a new prompt case
    version_payload = {
        "template": {
            "systemPrompt": "System instructions for greeting",
            "userPrompt": "Hello {{name}}",
            "format": "STANDARD",
        },
        "variables": [
            {"name": "name", "type": "string", "required": True}
        ],
        "author": "admin_user",
    }
    res_create = await client.post(
        "/v1/manager/prompt-cases/greet_user/versions", json=version_payload
    )
    assert res_create.status_code == 201
    v1_data = res_create.json()
    assert v1_data["versionNumber"] == 1
    assert v1_data["state"] == "DRAFT"
    assert v1_data["author"] == "admin_user"

    # 2. Publish version 1
    res_pub = await client.post(
        "/v1/manager/prompt-cases/greet_user/versions/1/publish"
    )
    assert res_pub.status_code == 200
    pub_data = res_pub.json()
    assert pub_data["state"] == "PUBLISHED"
    assert pub_data["versionHash"] is not None

    # 3. Create version 2 and publish -> version 1 becomes DEPRECATED
    res_create2 = await client.post(
        "/v1/manager/prompt-cases/greet_user/versions", json=version_payload
    )
    assert res_create2.status_code == 201

    res_pub2 = await client.post(
        "/v1/manager/prompt-cases/greet_user/versions/2/publish"
    )
    assert res_pub2.status_code == 200
    assert res_pub2.json()["state"] == "PUBLISHED"

    # 4. Get case details
    res_get = await client.get("/v1/manager/prompt-cases/greet_user")
    assert res_get.status_code == 200
    case_data = res_get.json()
    assert case_data["caseKey"] == "greet_user"
    assert case_data["publishedVersionNumber"] == 2
    # Verify version 1 is now deprecated
    v1_state = next(v["state"] for v in case_data["versions"] if v["versionNumber"] == 1)
    assert v1_state == "DEPRECATED"


@pytest.mark.asyncio
async def test_manager_api_endpoint_catalog(client):
    # Create entry
    entry_payload = {
        "operationId": "findMasters",
        "description": "Find active salon masters",
        "inputVariables": ["service_id!"],
        "outputVariables": ["master_ids"],
        "isReadOnly": True,
        "revision": "rev_ep_1",
    }
    res_post = await client.post("/v1/manager/endpoint-catalog", json=entry_payload)
    assert res_post.status_code == 201

    # List entries
    res_list = await client.get("/v1/manager/endpoint-catalog")
    assert res_list.status_code == 200
    items = res_list.json()
    assert any(i["operationId"] == "findMasters" for i in items)

    # Delete entry
    res_del = await client.delete("/v1/manager/endpoint-catalog/findMasters")
    assert res_del.status_code == 200


@pytest.mark.asyncio
async def test_manager_api_variable_catalog(client):
    # Create entry
    var_payload = {
        "variableId": "salon_branch_id",
        "type": "string",
        "description": "Unique branch identifier",
        "revision": "rev_var_1",
    }
    res_post = await client.post("/v1/manager/variable-catalog", json=var_payload)
    assert res_post.status_code == 201

    # List entries
    res_list = await client.get("/v1/manager/variable-catalog")
    assert res_list.status_code == 200
    items = res_list.json()
    assert any(i["variableId"] == "salon_branch_id" for i in items)

    # Delete entry
    res_del = await client.delete("/v1/manager/variable-catalog/salon_branch_id")
    assert res_del.status_code == 200


@pytest.mark.asyncio
async def test_localization_api_online_and_offline(client):
    # Online template resolve
    online_payload = {
        "caseKey": "appointment_reminder",
        "locale": "ru",
        "variables": {"client_name": "Алишер", "time": "15:00"},
    }
    res_on = await client.post("/v1/localization/online/resolve", json=online_payload)
    assert res_on.status_code == 200
    on_data = res_on.json()
    assert on_data["locale"] == "ru"
    assert "text" in on_data

    # Offline stateless translation generation (C-27)
    offline_payload = {
        "resourceKey": "ui.button.confirm",
        "targetLocale": "uz",
        "sourceText": "Подтвердить",
    }
    res_off = await client.post("/v1/localization/offline/generate", json=offline_payload)
    assert res_off.status_code == 200
    off_data = res_off.json()
    assert off_data["resourceKey"] == "ui.button.confirm"
    assert off_data["targetLocale"] == "uz"
    assert "generatedText" in off_data
