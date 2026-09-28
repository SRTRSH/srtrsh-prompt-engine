"""FastAPI routing for Resolver API and platform endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from srtrsh_prompt_engine.api.dto import (
    BoundVariableDTO,
    ExecutionRequestDTO,
    ExecutionResultDTO,
    ResolutionAndExecutionResponseDTO,
    ResolutionRequestDTO,
    ResolutionResultDTO,
)
from srtrsh_prompt_engine.domain.models import (
    ExecutionConfiguration,
    ProviderModel,
    SchemaContract,
)
from srtrsh_prompt_engine.domain.runtime import ExecutionRequest, ResolutionRequest
from srtrsh_prompt_engine.resolver.service import ResolverService


def create_resolver_router(resolver_service: ResolverService) -> APIRouter:
    router = APIRouter()

    @router.get("/health", status_code=status.HTTP_200_OK)
    async def health_check():
        return {"status": "ok"}

    @router.post(
        "/v1/resolver/resolve-and-execute",
        response_model=ResolutionAndExecutionResponseDTO,
        status_code=status.HTTP_200_OK,
    )
    async def resolve_and_execute_endpoint(dto: ResolutionRequestDTO):
        req = ResolutionRequest(
            prompt_identity=dto.prompt_identity,
            version_selector=dto.version_selector,
            raw_input=dto.raw_input,
            input=dto.input,
            runtime_context=dto.runtime_context,
            runtime_overrides=dto.runtime_overrides,
        )
        try:
            res = await resolver_service.resolve_and_execute(req)
        except KeyError as err:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(err)
            )
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)
            )

        bound_dto = [
            BoundVariableDTO(name=b.name, value=b.value, source=b.source.value)
            for b in res.bound_variables
        ]

        usage_dict = None
        if res.usage_telemetry:
            usage_dict = {
                "inputTokens": res.usage_telemetry.input_tokens,
                "outputTokens": res.usage_telemetry.output_tokens,
                "totalTokens": res.usage_telemetry.total_tokens,
                "latencyMs": res.usage_telemetry.latency_ms,
            }

        error_dict = None
        if res.error_outcome:
            error_dict = {
                "code": res.error_outcome.code,
                "message": res.error_outcome.message,
                "details": res.error_outcome.details,
            }

        return ResolutionAndExecutionResponseDTO(
            promptIdentity=res.prompt_identity,
            resolvedVersion=res.resolved_version,
            versionHash=res.version_hash,
            isReady=res.is_ready,
            boundVariables=bound_dto,
            missingVariables=res.missing_variables,
            executionId=res.execution_id,
            status=res.status.value if res.status else None,
            structuredOutput=res.structured_output,
            errorOutcome=error_dict,
            usageTelemetry=usage_dict,
        )

    @router.post(
        "/v1/resolver/resolve",
        response_model=ResolutionResultDTO,
        status_code=status.HTTP_200_OK,
    )
    async def resolve_endpoint(dto: ResolutionRequestDTO):
        req = ResolutionRequest(
            prompt_identity=dto.prompt_identity,
            version_selector=dto.version_selector,
            raw_input=dto.raw_input,
            input=dto.input,
            runtime_context=dto.runtime_context,
            runtime_overrides=dto.runtime_overrides,
        )
        try:
            res = await resolver_service.resolve(req)
        except KeyError as err:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(err)
            )
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)
            )

        bound_dto = [
            BoundVariableDTO(name=b.name, value=b.value, source=b.source.value)
            for b in res.bound_variables
        ]

        return ResolutionResultDTO(
            promptIdentity=res.prompt_identity,
            resolvedVersion=res.resolved_version,
            versionHash=res.version_hash,
            isReady=res.is_ready,
            boundVariables=bound_dto,
            missingVariables=res.missing_variables,
        )

    @router.post(
        "/v1/resolver/execute",
        response_model=ExecutionResultDTO,
        status_code=status.HTTP_200_OK,
    )
    async def execute_endpoint(dto: ExecutionRequestDTO):
        pm_data = dto.effective_configuration.get("providerModel")
        pm = None
        if pm_data:
            pm = ProviderModel(
                provider_id=pm_data.get("providerId", "mock"),
                model_name=pm_data.get("modelName", "default"),
            )

        cfg = ExecutionConfiguration(
            provider_model=pm,
            temperature=dto.effective_configuration.get("temperature"),
            timeout_duration=dto.effective_configuration.get("timeoutDuration"),
            max_output_tokens=dto.effective_configuration.get("maxOutputTokens"),
        )

        req = ExecutionRequest(
            execution_id=dto.execution_id or "",
            prompt_version_hash=dto.prompt_version_hash,
            hydrated_system_prompt=dto.hydrated_system_prompt,
            hydrated_user_prompt=dto.hydrated_user_prompt,
            resolved_context=dto.resolved_context,
            target_schema=SchemaContract(
                schema_type="JSON_SCHEMA",
                definition=dto.target_schema,
            ),
            effective_configuration=cfg,
        )

        try:
            res = await resolver_service.execute(req)
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)
            )

        usage_dict = {
            "inputTokens": res.usage_telemetry.input_tokens,
            "outputTokens": res.usage_telemetry.output_tokens,
            "totalTokens": res.usage_telemetry.total_tokens,
            "latencyMs": res.usage_telemetry.latency_ms,
        }

        error_dict = None
        if res.error_outcome:
            error_dict = {
                "code": res.error_outcome.code,
                "message": res.error_outcome.message,
                "details": res.error_outcome.details,
            }

        return ExecutionResultDTO(
            executionId=res.execution_id,
            promptVersionHash=res.prompt_version_hash,
            status=res.status.value,
            structuredOutput=res.structured_output,
            errorOutcome=error_dict,
            usageTelemetry=usage_dict,
        )

    return router
