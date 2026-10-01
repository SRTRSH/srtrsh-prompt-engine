"""Execution outcomes and interactive models.

Normative references:
WikiDocs/14-prompt-engine/07-execution-model/09-execution-outcomes.md
WikiDocs/14-prompt-engine/07-execution-model/10-execution-failure-semantics.md
WikiDocs/14-prompt-engine/14-api-contracts/02-runtime-api.md
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class ExecutionOutcomeType(str, Enum):
    """Terminal AI execution outcomes returned across Runtime API (Master section 59)."""

    COMPLETED = "COMPLETED"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    INTERACTIVE_RESPONSE = "INTERACTIVE_RESPONSE"
    HUMAN_FALLBACK = "HUMAN_FALLBACK"
    LIMIT_EXHAUSTED = "LIMIT_EXHAUSTED"
    TECHNICAL_ERROR = "TECHNICAL_ERROR"


class InteractiveOption(BaseModel):
    """Single selectable button option in an interactive response."""

    label: str
    result: Dict[str, Any]


class ExecutionOutcome(BaseModel):
    """Terminal execution outcome returned to the caller at the end of an attempt."""

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ExecutionOutcomeType
    text: Optional[str] = None
    options: List[InteractiveOption] = Field(default_factory=list)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_version: Optional[int] = None
    structured_state: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)
