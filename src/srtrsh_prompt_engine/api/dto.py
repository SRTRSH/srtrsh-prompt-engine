"""API Data Transfer Objects for the Prompt Engine Resolver API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BoundVariableDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    value: Any
    source: str


class ResolutionRequestDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt_identity: str = Field(alias="promptIdentity")
    version_selector: str = Field(default="ACTIVE", alias="versionSelector")
    raw_input: Optional[str] = Field(default=None, alias="rawInput")
    input: Dict[str, Any] = Field(default_factory=dict)
    runtime_context: Dict[str, Any] = Field(default_factory=dict, alias="runtimeContext")
    runtime_overrides: Dict[str, Any] = Field(default_factory=dict, alias="runtimeOverrides")


class ResolutionResultDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt_identity: str = Field(alias="promptIdentity")
    resolved_version: int = Field(alias="resolvedVersion")
    version_hash: str = Field(alias="versionHash")
    is_ready: bool = Field(alias="isReady")
    bound_variables: List[BoundVariableDTO] = Field(default_factory=list, alias="boundVariables")
    missing_variables: List[str] = Field(default_factory=list, alias="missingVariables")


class ExecutionRequestDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    execution_id: Optional[str] = Field(default=None, alias="executionId")
    prompt_version_hash: str = Field(alias="promptVersionHash")
    hydrated_system_prompt: str = Field(alias="hydratedSystemPrompt")
    hydrated_user_prompt: str = Field(alias="hydratedUserPrompt")
    resolved_context: Dict[str, Any] = Field(default_factory=dict, alias="resolvedContext")
    target_schema: Dict[str, Any] = Field(default_factory=dict, alias="targetSchema")
    effective_configuration: Dict[str, Any] = Field(default_factory=dict, alias="effectiveConfiguration")


class ExecutionResultDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    execution_id: str = Field(alias="executionId")
    prompt_version_hash: str = Field(alias="promptVersionHash")
    status: str
    structured_output: Optional[Dict[str, Any]] = Field(default=None, alias="structuredOutput")
    error_outcome: Optional[Dict[str, Any]] = Field(default=None, alias="errorOutcome")
    usage_telemetry: Optional[Dict[str, Any]] = Field(default=None, alias="usageTelemetry")


class ResolutionAndExecutionResponseDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt_identity: str = Field(alias="promptIdentity")
    resolved_version: int = Field(alias="resolvedVersion")
    version_hash: str = Field(alias="versionHash")
    is_ready: bool = Field(alias="isReady")
    bound_variables: List[BoundVariableDTO] = Field(default_factory=list, alias="boundVariables")
    missing_variables: List[str] = Field(default_factory=list, alias="missingVariables")
    execution_id: Optional[str] = Field(default=None, alias="executionId")
    status: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = Field(default=None, alias="structuredOutput")
    error_outcome: Optional[Dict[str, Any]] = Field(default=None, alias="errorOutcome")
    usage_telemetry: Optional[Dict[str, Any]] = Field(default=None, alias="usageTelemetry")
