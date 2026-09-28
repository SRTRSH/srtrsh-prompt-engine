"""Storage module."""

from srtrsh_prompt_engine.storage.repository import (
    InMemoryPromptRepository,
    PromptRepository,
)

__all__ = ["PromptRepository", "InMemoryPromptRepository"]
