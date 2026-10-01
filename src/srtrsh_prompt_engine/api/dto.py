"""API Data Transfer Objects for the Prompt Engine APIs (Resolver, Runtime, Manager, Localization)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Resolver API DTOs ---
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


# --- Runtime API DTOs (C-23) ---
class RuntimeExecuteRequestDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    conversation_id: str = Field(alias="conversationId")
    message: Optional[str] = None
    runtime_values: Dict[str, Any] = Field(default_factory=dict, alias="runtimeValues")
    business_references: Dict[str, Any] = Field(default_factory=dict, alias="businessReferences")
    case_key: str = Field(default="resolve_user_request", alias="caseKey")


class RuntimeInteractionRequestDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    conversation_id: str = Field(alias="conversationId")
    interaction_result: Dict[str, Any] = Field(alias="interactionResult")
    case_key: str = Field(default="resolve_user_request", alias="caseKey")


class InteractiveOptionDTO(BaseModel):
    label: str
    result: Dict[str, Any]


class RuntimeOutcomeDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    execution_id: str = Field(alias="executionId")
    status: str
    text: Optional[str] = None
    options: List[InteractiveOptionDTO] = Field(default_factory=list)
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId")
    session_version: Optional[int] = Field(default=None, alias="sessionVersion")
    structured_state: Dict[str, Any] = Field(default_factory=dict, alias="structuredState")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    telemetry: Dict[str, Any] = Field(default_factory=dict)


# --- Manager API DTOs (C-21, C-24) ---
class PromptVersionDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    version_number: int = Field(alias="versionNumber")
    state: str
    version_hash: Optional[str] = Field(default=None, alias="versionHash")
    created_at: str = Field(alias="createdAt")
    published_at: Optional[str] = Field(default=None, alias="publishedAt")
    author: Optional[str] = None


class PromptCaseDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_key: str = Field(alias="caseKey")
    description: Optional[str] = None
    active_version_reference: Optional[int] = Field(default=None, alias="activeVersionReference")
    published_version_number: Optional[int] = Field(default=None, alias="publishedVersionNumber")
    versions: List[PromptVersionDTO] = Field(default_factory=list)


class CreatePromptVersionDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    template: Dict[str, Any]
    variables: List[Dict[str, Any]] = Field(default_factory=list)
    schema_contract: Optional[Dict[str, Any]] = Field(default=None, alias="schemaContract")
    execution_configuration: Optional[Dict[str, Any]] = Field(default=None, alias="executionConfiguration")
    override_policy: Optional[Dict[str, Any]] = Field(default=None, alias="overridePolicy")
    author: Optional[str] = None


class EndpointCatalogEntryDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    operation_id: str = Field(alias="operationId")
    description: str
    human_description: Optional[str] = Field(default=None, alias="humanDescription")
    input_variables: List[str] = Field(default_factory=list, alias="inputVariables")
    output_variables: List[str] = Field(default_factory=list, alias="outputVariables")
    is_read_only: bool = Field(default=True, alias="isReadOnly")
    execution_constraints: Dict[str, Any] = Field(default_factory=dict, alias="executionConstraints")
    revision: str = "rev_1"


class VariableCatalogEntryDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    variable_id: str = Field(alias="variableId")
    type: str
    description: str
    revision: str = "rev_1"


# --- Localization API DTOs (C-27) ---
class OnlineResolveRequestDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_key: str = Field(alias="caseKey")
    locale: str
    variables: Dict[str, Any] = Field(default_factory=dict)


class OnlineResolveResponseDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    locale: str
    case_key: str = Field(alias="caseKey")
    resolved_from: str = Field(alias="resolvedFrom")  # "STORED_TEMPLATE" or "GENERATED"


class OfflineGenerateRequestDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resource_key: str = Field(alias="resourceKey")
    target_locale: str = Field(alias="targetLocale")
    source_text: str = Field(alias="sourceText")
    variables: Dict[str, Any] = Field(default_factory=dict)


class OfflineGenerateResponseDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resource_key: str = Field(alias="resourceKey")
    target_locale: str = Field(alias="targetLocale")
    generated_text: str = Field(alias="generatedText")
