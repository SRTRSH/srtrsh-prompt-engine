"""Resolver Service coordinating Data Plane resolution and execution."""

from copy import deepcopy
from typing import Optional
import uuid

from srtrsh_prompt_engine.domain.models import ExecutionConfiguration, VersionState
from srtrsh_prompt_engine.domain.runtime import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ResolutionAndExecutionResponse,
    ResolutionRequest,
    ResolutionResult,
    TechnicalErrorOutcome,
)
from srtrsh_prompt_engine.resolver.binder import bind_variables
from srtrsh_prompt_engine.resolver.hydrator import hydrate_template
from srtrsh_prompt_engine.resolver.validator import validate_schema
from srtrsh_prompt_engine.spi.provider import (
    HydratedPrompt,
    ProviderExecutionRequest,
)
from srtrsh_prompt_engine.spi.registry import ProviderRegistry
from srtrsh_prompt_engine.storage.repository import PromptRepository


class ResolverService:
    """Core Data Plane engine for prompt resolution and execution."""

    def __init__(
        self,
        repository: PromptRepository,
        provider_registry: ProviderRegistry,
    ) -> None:
        self.repository = repository
        self.provider_registry = provider_registry

    async def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        prompt = self.repository.get_by_identity(request.prompt_identity)
        if not prompt:
            raise KeyError(f"Prompt '{request.prompt_identity}' not found.")

        # Resolve version selector
        if request.version_selector == "ACTIVE":
            version = prompt.get_active_version()
            if not version:
                raise ValueError(
                    f"No active version found for prompt '{request.prompt_identity}'."
                )
        else:
            try:
                version_number = int(request.version_selector)
            except ValueError:
                raise ValueError(
                    f"Invalid version selector '{request.version_selector}'."
                )
            version = prompt.get_version(version_number)
            if not version:
                raise KeyError(
                    f"Version {version_number} not found for prompt '{request.prompt_identity}'."
                )

        if version.state != VersionState.PUBLISHED:
            raise ValueError(
                f"Version {version.version_number} is in state {version.state}. "
                "Only PUBLISHED versions can be resolved for execution."
            )

        bound_variables, missing_variables = bind_variables(version, request)
        is_ready = len(missing_variables) == 0

        return ResolutionResult(
            prompt_identity=prompt.identity,
            resolved_version=version.version_number,
            version_hash=version.version_hash or "",
            is_ready=is_ready,
            bound_variables=bound_variables,
            missing_variables=[m.name for m in missing_variables],
        )

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        cfg = request.effective_configuration
        provider_id = cfg.provider_model.provider_id if cfg.provider_model else "mock"
        model_name = cfg.provider_model.model_name if cfg.provider_model else "default"

        provider = self.provider_registry.get_or_default(provider_id)
        if not provider:
            return ExecutionResult(
                execution_id=request.execution_id,
                prompt_version_hash=request.prompt_version_hash,
                status=ExecutionStatus.TECHNICAL_FAILURE,
                error_outcome=TechnicalErrorOutcome(
                    code="PROVIDER_NOT_FOUND",
                    message=f"Provider '{provider_id}' is not registered.",
                ),
            )

        parameters = {
            "temperature": cfg.temperature,
            "timeout_duration": cfg.timeout_duration,
            "max_output_tokens": cfg.max_output_tokens,
        }

        prov_req = ProviderExecutionRequest(
            execution_id=request.execution_id,
            prompt_version_hash=request.prompt_version_hash,
            provider=provider_id,
            model=model_name,
            prompt=HydratedPrompt(
                system_prompt=request.hydrated_system_prompt,
                user_prompt=request.hydrated_user_prompt,
            ),
            target_schema=request.target_schema.definition,
            parameters=parameters,
            metadata=request.resolved_context,
        )

        resp = await provider.execute(prov_req)

        if resp.status != "SUCCESS":
            return ExecutionResult(
                execution_id=request.execution_id,
                prompt_version_hash=request.prompt_version_hash,
                status=ExecutionStatus.TECHNICAL_FAILURE,
                error_outcome=TechnicalErrorOutcome(
                    code="PROVIDER_ERROR",
                    message=resp.error_message or "External provider call failed.",
                ),
                usage_telemetry=resp.usage,
            )

        # Authoritative Schema Validation
        if request.target_schema.definition:
            output_data = resp.output or {}
            is_valid, err_msg = validate_schema(
                output_data, request.target_schema.definition
            )
            if not is_valid:
                return ExecutionResult(
                    execution_id=request.execution_id,
                    prompt_version_hash=request.prompt_version_hash,
                    status=ExecutionStatus.TECHNICAL_FAILURE,
                    error_outcome=TechnicalErrorOutcome(
                        code="SCHEMA_VALIDATION_ERROR",
                        message=f"Output does not conform to schema contract: {err_msg}",
                    ),
                    usage_telemetry=resp.usage,
                )

        return ExecutionResult(
            execution_id=request.execution_id,
            prompt_version_hash=request.prompt_version_hash,
            status=ExecutionStatus.SUCCESS,
            structured_output=resp.output,
            usage_telemetry=resp.usage,
        )

    async def resolve_and_execute(
        self, request: ResolutionRequest
    ) -> ResolutionAndExecutionResponse:
        resolution = await self.resolve(request)

        # Hard readiness gate: if not ready, NEVER call provider
        if not resolution.is_ready:
            return ResolutionAndExecutionResponse(
                prompt_identity=resolution.prompt_identity,
                resolved_version=resolution.resolved_version,
                version_hash=resolution.version_hash,
                is_ready=False,
                bound_variables=resolution.bound_variables,
                missing_variables=resolution.missing_variables,
            )

        # Ready: load immutable PromptVersion
        prompt = self.repository.get_by_identity(resolution.prompt_identity)
        if not prompt:
            raise KeyError(f"Prompt '{resolution.prompt_identity}' not found.")
        version = prompt.get_version(resolution.resolved_version)
        if not version:
            raise KeyError(
                f"Version {resolution.resolved_version} not found for prompt '{resolution.prompt_identity}'."
            )

        # Enforce override policy
        allowed_overrides = version.definition.override_policy.allowed_parameters
        for param in request.runtime_overrides:
            if param not in allowed_overrides:
                raise ValueError(
                    f"Runtime override '{param}' is not permitted by PromptVersion policy."
                )

        # Construct effective configuration
        effective_cfg = deepcopy(version.definition.execution_configuration)
        if "temperature" in request.runtime_overrides:
            effective_cfg.temperature = float(request.runtime_overrides["temperature"])
        if "max_output_tokens" in request.runtime_overrides:
            effective_cfg.max_output_tokens = int(
                request.runtime_overrides["max_output_tokens"]
            )
        if "timeout_duration" in request.runtime_overrides:
            effective_cfg.timeout_duration = int(
                request.runtime_overrides["timeout_duration"]
            )

        # Prompt hydration
        bound_dict = {b.name: b.value for b in resolution.bound_variables}
        hydrated_system = hydrate_template(
            template_text=version.definition.template.system_prompt,
            bound_variables=bound_dict,
            raw_input=request.raw_input or "",
            runtime_context=request.runtime_context,
        )
        hydrated_user = hydrate_template(
            template_text=version.definition.template.user_prompt,
            bound_variables=bound_dict,
            raw_input=request.raw_input or "",
            runtime_context=request.runtime_context,
        )

        exec_request = ExecutionRequest(
            execution_id=str(uuid.uuid4()),
            prompt_version_hash=resolution.version_hash,
            hydrated_system_prompt=hydrated_system,
            hydrated_user_prompt=hydrated_user,
            resolved_context=request.runtime_context,
            target_schema=version.definition.schema_contract,
            effective_configuration=effective_cfg,
        )

        exec_result = await self.execute(exec_request)

        return ResolutionAndExecutionResponse(
            prompt_identity=resolution.prompt_identity,
            resolved_version=resolution.resolved_version,
            version_hash=resolution.version_hash,
            is_ready=True,
            bound_variables=resolution.bound_variables,
            missing_variables=[],
            execution_id=exec_result.execution_id,
            status=exec_result.status,
            structured_output=exec_result.structured_output,
            error_outcome=exec_result.error_outcome,
            usage_telemetry=exec_result.usage_telemetry,
        )
