"""Prompt Engine core domain models, aggregates, and value objects.

Normative references:
WikiDocs/14-prompt-engine/03-domain-model/01-prompt-case.md
WikiDocs/14-prompt-engine/03-domain-model/02-prompt-version.md
WikiDocs/14-prompt-engine/03-domain-model/03-organization-system-prompt.md
WikiDocs/14-prompt-engine/03-domain-model/04-endpoint-catalog.md
WikiDocs/14-prompt-engine/03-domain-model/05-variable-catalog.md
WikiDocs/14-prompt-engine/03-domain-model/06-ai-session.md
WikiDocs/14-prompt-engine/03-domain-model/07-conversation-history.md
WikiDocs/14-prompt-engine/03-domain-model/08-session-state.md
WikiDocs/14-prompt-engine/03-domain-model/09-business-references.md
WikiDocs/14-prompt-engine/04-identities-versioning-and-invariants/04-invariants.md
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/06-session-concurrency.md
WikiDocs/14-prompt-engine/10-localization-and-templates/
WikiDocs/14-prompt-engine/15-persistence/
"""

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set
import uuid

from pydantic import BaseModel, Field, model_validator

from srtrsh_prompt_engine.canonical.hasher import (
    build_canonical_payload,
    compute_prompt_version_hash,
)


class VersionState(str, Enum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    DEPRECATED = "DEPRECATED"


class VariableType(str, Enum):
    """Closed enum of supported Variable types.

    Normative: string, integer, number, boolean, datetime, array, object.
    """

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"

    @classmethod
    def _missing_(cls, value: object) -> Optional["VariableType"]:
        if isinstance(value, str):
            val_lower = value.strip().lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


class VariableSource(str, Enum):
    RUNTIME_INPUT = "RUNTIME_INPUT"
    CONTEXT_PAYLOAD = "CONTEXT_PAYLOAD"
    DEFAULT_VALUE = "DEFAULT_VALUE"


class MissingReason(str, Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    NULL_VALUE = "NULL_VALUE"
    TYPE_MISMATCH = "TYPE_MISMATCH"


class SessionConcurrencyError(Exception):
    """Raised when an optimistic locking check fails on an AI Session update."""


class Template(BaseModel):
    system_prompt: str
    user_prompt: str
    format: str = "STANDARD"


class VariableDefinition(BaseModel):
    name: str
    type: str = "string"
    required: bool = True
    default_value: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_var_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "type" in data:
            raw_type = data["type"]
            if isinstance(raw_type, str):
                vt = VariableType(raw_type)
                data["type"] = vt.value
        return data

    @model_validator(mode="after")
    def validate_default_and_required(self) -> "VariableDefinition":
        if self.required and self.default_value is not None:
            raise ValueError(f"Required variable '{self.name}' must not have a defaultValue.")
        return self


class SchemaContract(BaseModel):
    schema_type: str = "JSON_SCHEMA"
    definition: Dict[str, Any] = Field(default_factory=dict)


class ProviderModel(BaseModel):
    provider_id: str
    model_name: str


class ExecutionConfiguration(BaseModel):
    provider_model: Optional[ProviderModel] = None
    temperature: Optional[float] = None
    timeout_duration: Optional[int] = None  # in milliseconds
    max_output_tokens: Optional[int] = None


class RuntimeOverridePolicy(BaseModel):
    allowed_parameters: Set[str] = Field(default_factory=set)


class HashedVersionDefinition(BaseModel):
    template: Template
    variables: List[VariableDefinition] = Field(default_factory=list)
    schema_contract: SchemaContract
    execution_configuration: ExecutionConfiguration = Field(
        default_factory=ExecutionConfiguration
    )
    override_policy: RuntimeOverridePolicy = Field(
        default_factory=RuntimeOverridePolicy
    )

    def to_canonical_payload(self) -> Dict[str, Any]:
        template_dict = {
            "format": self.template.format,
            "systemPrompt": self.template.system_prompt,
            "userPrompt": self.template.user_prompt,
        }
        vars_list = [
            {
                "defaultValue": v.default_value,
                "name": v.name,
                "required": v.required,
                "type": v.type,
            }
            for v in self.variables
        ]
        schema_dict = {
            "definition": self.schema_contract.definition,
            "schemaType": self.schema_contract.schema_type,
        }
        exec_cfg_dict: Dict[str, Any] = {
            "maxOutputTokens": self.execution_configuration.max_output_tokens,
            "providerModel": (
                {
                    "modelName": self.execution_configuration.provider_model.model_name,
                    "providerId": self.execution_configuration.provider_model.provider_id,
                }
                if self.execution_configuration.provider_model
                else None
            ),
            "temperature": self.execution_configuration.temperature,
            "timeoutDuration": self.execution_configuration.timeout_duration,
        }
        override_dict = {
            "allowedParameters": list(self.override_policy.allowed_parameters)
        }
        return build_canonical_payload(
            template=template_dict,
            variables=vars_list,
            schema_contract=schema_dict,
            execution_configuration=exec_cfg_dict,
            override_policy=override_dict,
        )

    def compute_hash(self) -> str:
        payload = self.to_canonical_payload()
        return compute_prompt_version_hash(payload)


class BoundVariable(BaseModel):
    name: str
    value: Any
    source: VariableSource


class MissingVariable(BaseModel):
    name: str
    reason: MissingReason = MissingReason.NOT_PROVIDED


class PromptVersion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_key: str = ""
    version_number: int
    state: VersionState = VersionState.DRAFT
    definition: HashedVersionDefinition
    version_hash: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: Optional[datetime] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "definition"
            and hasattr(self, "state")
            and self.state == VersionState.PUBLISHED
        ):
            raise ValueError("Cannot modify definition of a PUBLISHED version. Published versions are immutable.")
        super().__setattr__(name, value)

    def publish(self) -> None:
        """Publish this version, freezing the definition and computing PromptVersionHash."""
        if self.state == VersionState.DEPRECATED:
            raise ValueError("Cannot publish a deprecated version.")
        if self.state == VersionState.PUBLISHED:
            return  # Idempotent
        self.version_hash = self.definition.compute_hash()
        self.state = VersionState.PUBLISHED
        self.published_at = datetime.now(timezone.utc)

    def deprecate(self) -> None:
        """Deprecate this version."""
        if self.state == VersionState.DRAFT:
            raise ValueError("Cannot deprecate a draft version.")
        self.state = VersionState.DEPRECATED


class PromptCase(BaseModel):
    """Prompt Case aggregate root. Identified by stable immutable caseKey."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_key: str
    description: Optional[str] = None
    active_version_reference: Optional[int] = None
    versions: Dict[int, PromptVersion] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __init__(self, **data: Any):
        if "identity" in data and "case_key" not in data and "caseKey" not in data:
            data["case_key"] = data.pop("identity")
        elif "caseKey" in data and "case_key" not in data:
            data["case_key"] = data.pop("caseKey")
        super().__init__(**data)

    @property
    def identity(self) -> str:
        """Backward-compatibility alias for case_key."""
        return self.case_key

    @identity.setter
    def identity(self, val: str) -> None:
        self.case_key = val

    @property
    def published_version_number(self) -> Optional[int]:
        """Returns the version number of the currently active published version."""
        return self.active_version_reference

    def create_version(
        self, definition: HashedVersionDefinition, author: Optional[str] = None
    ) -> PromptVersion:
        next_version_number = max(self.versions.keys(), default=0) + 1
        version = PromptVersion(
            case_key=self.case_key,
            version_number=next_version_number,
            definition=definition,
            state=VersionState.DRAFT,
            author=author,
        )
        self.versions[next_version_number] = version
        self.updated_at = datetime.now(timezone.utc)
        return version

    def publish_version(self, version_number: int) -> PromptVersion:
        if version_number not in self.versions:
            raise ValueError(
                f"Version {version_number} not found for prompt case '{self.case_key}'."
            )
        target_version = self.versions[version_number]
        if target_version.state == VersionState.DEPRECATED:
            raise ValueError("Cannot publish a deprecated version.")
        if target_version.state == VersionState.PUBLISHED:
            return target_version

        # Invariant AMB-01: At most one version of a given Prompt Case may be in PUBLISHED status at any time.
        # Transitioning a draft to PUBLISHED atomically transitions any previously PUBLISHED version to DEPRECATED.
        for existing in self.versions.values():
            if existing.version_number != version_number and existing.state == VersionState.PUBLISHED:
                existing.deprecate()

        target_version.publish()
        self.active_version_reference = version_number
        self.updated_at = datetime.now(timezone.utc)
        return target_version

    def activate_version(self, version_number: int) -> None:
        if version_number not in self.versions:
            raise ValueError(
                f"Version {version_number} not found for prompt case '{self.case_key}'."
            )
        version = self.versions[version_number]
        if version.state != VersionState.PUBLISHED:
            raise ValueError(
                f"Cannot activate version {version_number} with state {version.state}. "
                "Only PUBLISHED versions can be active."
            )
        self.active_version_reference = version_number
        self.updated_at = datetime.now(timezone.utc)

    def deprecate_version(self, version_number: int) -> None:
        if version_number not in self.versions:
            raise ValueError(
                f"Version {version_number} not found for prompt case '{self.case_key}'."
            )
        version = self.versions[version_number]
        version.deprecate()
        if self.active_version_reference == version_number:
            self.active_version_reference = None
        self.updated_at = datetime.now(timezone.utc)

    def get_active_version(self) -> Optional[PromptVersion]:
        if self.active_version_reference is None:
            return None
        return self.versions.get(self.active_version_reference)

    def get_published_version(self) -> Optional[PromptVersion]:
        return self.get_active_version()

    def get_version(self, version_number: int) -> Optional[PromptVersion]:
        return self.versions.get(version_number)


Prompt = PromptCase


class EndpointCatalogEntry(BaseModel):
    """Structured registry entry for an operation the AI is allowed to request."""

    operation_id: str
    description: str  # AI-oriented description
    human_description: Optional[str] = None
    input_variables: List[str] = Field(default_factory=list)  # e.g. ["master_id!", "service_id!"]
    output_variables: List[str] = Field(default_factory=list)
    is_read_only: bool = True  # Safety boundary: only READ operations in MVP (C-11)
    execution_constraints: Dict[str, Any] = Field(default_factory=dict)
    revision: str = "rev_1"

    @model_validator(mode="after")
    def validate_safety_boundary(self) -> "EndpointCatalogEntry":
        if not self.is_read_only:
            raise ValueError(
                f"Write operation '{self.operation_id}' is not permitted. "
                "Only READ operations are allowed in Prompt Engine MVP (C-11)."
            )
        return self


class VariableCatalogEntry(BaseModel):
    """Structured registry entry for a variable exposed to LLM reasoning."""

    variable_id: str
    type: VariableType
    description: str
    revision: str = "rev_1"

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "type" in data:
            raw_type = data["type"]
            if isinstance(raw_type, str):
                try:
                    data["type"] = VariableType(raw_type)
                except ValueError:
                    allowed = [m.value for m in VariableType]
                    raise ValueError(
                        f"Invalid variable type '{raw_type}'. Must be one of: {allowed}"
                    )
        return data


class AISession(BaseModel):
    """Persistent AI Session aggregate. Tracks state, references, and monotonic session version."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    session_version: int = 1
    session_state: Dict[str, Any] = Field(default_factory=dict)
    business_references: Dict[str, Any] = Field(default_factory=dict)
    interaction_results: List[Dict[str, Any]] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_session_version(self) -> "AISession":
        if self.session_version is None or self.session_version < 1:
            raise ValueError("session_version must be a non-null positive integer >= 1.")
        return self


class SessionMessage(BaseModel):
    """Durable conversation history message."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    conversation_id: str
    role: str  # "customer", "assistant", "system"
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OnlineTemplate(BaseModel):
    """Dynamic localized online template owned by Prompt Engine."""

    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_key: str
    locale: str
    template_text: str
    required_variables: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def extract_variables(self) -> Set[str]:
        """Extract all $variable$ or {{variable}} references from template_text."""
        vars_dollar = set(re.findall(r"\$([a-zA-Z0-9_]+)\$", self.template_text))
        vars_mustache = set(re.findall(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", self.template_text))
        return vars_dollar | vars_mustache

    def validate_variables(self, provided: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate that all required variables are provided and have non-null values."""
        missing = [v for v in self.required_variables if v not in provided or provided[v] is None]
        return len(missing) == 0, missing

    def render(self, variables: Dict[str, Any]) -> str:
        """Deterministically substitute variables into template_text."""
        valid, missing = self.validate_variables(variables)
        if not valid:
            raise ValueError(f"Cannot render template: missing variables: {missing}")

        result = self.template_text
        for var_name, val in variables.items():
            str_val = str(val) if val is not None else ""
            result = result.replace(f"${var_name}$", str_val)
            pattern = r"\{\{\s*" + re.escape(var_name) + r"\s*\}\}"
            result = re.sub(pattern, str_val, result)
        return result


class AuditRecord(BaseModel):
    """Durable execution audit record for reproducibility and compliance."""

    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    case_key: str
    prompt_version_number: int
    prompt_version_hash: str
    endpoint_catalog_revision: Optional[str] = None
    variable_catalog_revision: Optional[str] = None
    effective_configuration: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
