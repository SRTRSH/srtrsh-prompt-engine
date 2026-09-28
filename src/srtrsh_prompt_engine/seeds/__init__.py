"""Seeds module."""

from srtrsh_prompt_engine.seeds.system_bot import (
    DIRECT_BOOKING_REQUEST_JSON_SCHEMA,
    create_system_bot_master_intent_prompt,
)

__all__ = [
    "DIRECT_BOOKING_REQUEST_JSON_SCHEMA",
    "create_system_bot_master_intent_prompt",
]
