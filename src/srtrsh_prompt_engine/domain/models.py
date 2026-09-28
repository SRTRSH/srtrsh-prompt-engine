"""Prompt Engine core domain models, aggregates, and value objects.

Normative reference:
WikiDocs/14-prompt-engine/02-domain-model/01-entities-and-templates.md
WikiDocs/14-prompt-engine/02-domain-model/02-invariants.md
WikiDocs/14-prompt-engine/02-domain-model/03-identity-and-versioning.md
"""

from datetime import datetime, timezone
from enum import Enum
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
    STRING = "STRING"
    INTEGER = "INTEGER"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    DATETIME = "DATETIME"
    ARRAY = "ARRAY"
    OBJECT = "OBJECT"


class VariableSource(str, Enum):
    RUNTIME_INPUT = "RUNTIME_INPUT"
    CONTEXT_PAYLOAD = "CONTEXT_PAYLOAD"
    DEFAULT_VALUE = "DEFAULT_VALUE"


class MissingReason(str, Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    NULL_VALUE = "NULL_VALUE"
    TYPE_MISMATCH = "TYPE_MISMATCH"


class Template(BaseModel):
    system_prompt: str
    user_prompt: str
    format: str = "STANDARD"


class VariableDefinition(BaseModel):
    name: str
    type: str = "STRING"
    required: bool = True
    default_value: Optional[Any] = None

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
    version_number: int
    state: VersionState = VersionState.DRAFT
    definition: HashedVersionDefinition
    version_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: Optional[datetime] = None

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


class Prompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    identity: str
    active_version_reference: Optional[int] = None
    versions: Dict[int, PromptVersion] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def create_version(self, definition: HashedVersionDefinition) -> PromptVersion:
        next_version_number = max(self.versions.keys(), default=0) + 1
        version = PromptVersion(
            version_number=next_version_number,
            definition=definition,
            state=VersionState.DRAFT,
        )
        self.versions[next_version_number] = version
        self.updated_at = datetime.now(timezone.utc)
        return version

    def publish_version(self, version_number: int) -> PromptVersion:
        if version_number not in self.versions:
            raise ValueError(f"Version {version_number} not found for prompt '{self.identity}'.")
        version = self.versions[version_number]
        version.publish()
        self.updated_at = datetime.now(timezone.utc)
        return version

    def activate_version(self, version_number: int) -> None:
        if version_number not in self.versions:
            raise ValueError(f"Version {version_number} not found for prompt '{self.identity}'.")
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
            raise ValueError(f"Version {version_number} not found for prompt '{self.identity}'.")
        version = self.versions[version_number]
        version.deprecate()
        if self.active_version_reference == version_number:
            self.active_version_reference = None
        self.updated_at = datetime.now(timezone.utc)

    def get_active_version(self) -> Optional[PromptVersion]:
        if self.active_version_reference is None:
            return None
        return self.versions.get(self.active_version_reference)

    def get_version(self, version_number: int) -> Optional[PromptVersion]:
        return self.versions.get(version_number)
