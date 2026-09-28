"""Data Plane transient runtime models for Resolution and Execution.

Normative reference:
WikiDocs/14-prompt-engine/02-domain-model/01-entities-and-templates.md
WikiDocs/14-prompt-engine/03-api-contracts/02-resolver-api.md
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from srtrsh_prompt_engine.domain.models import (
    BoundVariable,
    ExecutionConfiguration,
    SchemaContract,
)


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


class ResolutionRequest(BaseModel):
    prompt_identity: str
    version_selector: str = "ACTIVE"
    raw_input: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    runtime_context: Dict[str, Any] = Field(default_factory=dict)
    runtime_overrides: Dict[str, Any] = Field(default_factory=dict)


class ResolutionResult(BaseModel):
    prompt_identity: str
    resolved_version: int
    version_hash: str
    is_ready: bool
    bound_variables: List[BoundVariable] = Field(default_factory=list)
    missing_variables: List[str] = Field(default_factory=list)


class ExecutionRequest(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_version_hash: str
    hydrated_system_prompt: str
    hydrated_user_prompt: str
    resolved_context: Dict[str, Any] = Field(default_factory=dict)
    target_schema: SchemaContract
    effective_configuration: ExecutionConfiguration


class ExecutionUsageTelemetry(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class TechnicalErrorOutcome(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ExecutionResult(BaseModel):
    execution_id: str
    prompt_version_hash: str
    status: ExecutionStatus
    structured_output: Optional[Dict[str, Any]] = None
    error_outcome: Optional[TechnicalErrorOutcome] = None
    usage_telemetry: ExecutionUsageTelemetry = Field(
        default_factory=ExecutionUsageTelemetry
    )


class ResolutionAndExecutionResponse(BaseModel):
    prompt_identity: str
    resolved_version: int
    version_hash: str
    is_ready: bool
    bound_variables: List[BoundVariable] = Field(default_factory=list)
    missing_variables: List[str] = Field(default_factory=list)
    # Populated only if is_ready is True
    execution_id: Optional[str] = None
    status: Optional[ExecutionStatus] = None
    structured_output: Optional[Dict[str, Any]] = None
    error_outcome: Optional[TechnicalErrorOutcome] = None
    usage_telemetry: Optional[ExecutionUsageTelemetry] = None
