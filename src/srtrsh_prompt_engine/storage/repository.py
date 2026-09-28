"""Repository abstraction and in-memory implementation for Prompts."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from srtrsh_prompt_engine.domain.models import Prompt


class PromptRepository(ABC):
    """Abstract interface for Prompt aggregate persistence."""

    @abstractmethod
    def get_by_identity(self, identity: str) -> Optional[Prompt]:
        """Retrieve Prompt aggregate by its logical identity."""
        ...

    @abstractmethod
    def save(self, prompt: Prompt) -> None:
        """Persist or update Prompt aggregate."""
        ...


class InMemoryPromptRepository(PromptRepository):
    """In-memory storage for prompts and initial seed definitions."""

    def __init__(self) -> None:
        self._prompts_by_identity: Dict[str, Prompt] = {}

    def get_by_identity(self, identity: str) -> Optional[Prompt]:
        return self._prompts_by_identity.get(identity)

    def save(self, prompt: Prompt) -> None:
        self._prompts_by_identity[prompt.identity] = prompt
