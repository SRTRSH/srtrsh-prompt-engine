"""Seed definition for the canonical system_bot_master_intent Prompt.

This prompt powers the System Bot entry point for Masters creating DirectBookingRequest operations.
Normative reference:
WikiDocs/11-communication-channels/bot.md
WikiDocs/14-prompt-engine/02-domain-model/01-entities-and-templates.md
"""

from typing import Any, Dict

from srtrsh_prompt_engine.domain.models import (
    ExecutionConfiguration,
    HashedVersionDefinition,
    Prompt,
    ProviderModel,
    RuntimeOverridePolicy,
    SchemaContract,
    Template,
    VariableDefinition,
)

DIRECT_BOOKING_REQUEST_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DirectBookingRequest",
    "type": "object",
    "properties": {
        "master_id": {
            "type": "string",
            "format": "uuid",
        },
        "customer_id": {
            "type": "string",
            "format": "uuid",
        },
        "workplace_id": {
            "type": "string",
            "format": "uuid",
        },
        "start_time": {
            "type": "string",
            "format": "date-time",
        },
        "end_time": {
            "type": "string",
            "format": "date-time",
        },
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string",
                        "format": "uuid",
                    },
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
                "required": ["service_id", "quantity"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "master_id",
        "customer_id",
        "workplace_id",
        "start_time",
        "end_time",
        "items",
    ],
    "additionalProperties": False,
}


def create_system_bot_master_intent_prompt() -> Prompt:
    """Build and publish version 1 of the canonical system_bot_master_intent prompt."""
    prompt = Prompt(identity="system_bot_master_intent")

    template = Template(
        format="STANDARD",
        system_prompt=(
            "You are the SRTRSH System Bot intent engine. Transform communication channel "
            "requests from Masters into structured direct booking requests according to the "
            "DirectBookingRequest schema. Only return valid JSON complying strictly with the "
            "target schema. Do not include commentary or markdown fences."
        ),
        user_prompt=(
            "master_id: {{master_id}}\n"
            "customer_id: {{customer_id}}\n"
            "workplace_id: {{workplace_id}}\n"
            "start_time: {{start_time}}\n"
            "end_time: {{end_time}}\n"
            "items: {{items}}\n"
            "raw_input: {{raw_input}}"
        ),
    )

    variables = [
        VariableDefinition(name="master_id", type="STRING", required=True),
        VariableDefinition(name="customer_id", type="STRING", required=True),
        VariableDefinition(name="workplace_id", type="STRING", required=True),
        VariableDefinition(name="start_time", type="DATETIME", required=True),
        VariableDefinition(name="end_time", type="DATETIME", required=True),
        VariableDefinition(name="items", type="ARRAY", required=True),
        VariableDefinition(
            name="raw_input", type="STRING", required=False, default_value=""
        ),
    ]

    schema_contract = SchemaContract(
        schema_type="JSON_SCHEMA",
        definition=DIRECT_BOOKING_REQUEST_JSON_SCHEMA,
    )

    exec_config = ExecutionConfiguration(
        provider_model=ProviderModel(provider_id="mock", model_name="default"),
        temperature=0.0,
        timeout_duration=30000,
        max_output_tokens=2048,
    )

    override_policy = RuntimeOverridePolicy(
        allowed_parameters={"temperature", "max_output_tokens", "timeout_duration"}
    )

    definition = HashedVersionDefinition(
        template=template,
        variables=variables,
        schema_contract=schema_contract,
        execution_configuration=exec_config,
        override_policy=override_policy,
    )

    v1 = prompt.create_version(definition)
    prompt.publish_version(v1.version_number)
    prompt.activate_version(v1.version_number)
    return prompt
