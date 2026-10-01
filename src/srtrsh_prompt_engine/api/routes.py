"""FastAPI routing for Runtime API, Manager API, Localization API, and Resolver API."""

from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, HTTPException, status

from srtrsh_prompt_engine.api.dto import (
    BoundVariableDTO,
    CreatePromptVersionDTO,
    EndpointCatalogEntryDTO,
    ExecutionRequestDTO,
    ExecutionResultDTO,
    InteractiveOptionDTO,
    OfflineGenerateRequestDTO,
    OfflineGenerateResponseDTO,
    OnlineResolveRequestDTO,
    OnlineResolveResponseDTO,
    PromptCaseDTO,
    PromptVersionDTO,
    ResolutionAndExecutionResponseDTO,
    ResolutionRequestDTO,
    ResolutionResultDTO,
    RuntimeExecuteRequestDTO,
    RuntimeInteractionRequestDTO,
    RuntimeOutcomeDTO,
    VariableCatalogEntryDTO,
)
from srtrsh_prompt_engine.domain.models import (
    EndpointCatalogEntry,
    ExecutionConfiguration,
    HashedVersionDefinition,
    OnlineTemplate,
    PromptCase,
    ProviderModel,
    RuntimeOverridePolicy,
    SchemaContract,
    Template,
    VariableCatalogEntry,
    VariableDefinition,
    VariableType,
)
from srtrsh_prompt_engine.domain.runtime import ExecutionRequest, ResolutionRequest
from srtrsh_prompt_engine.execution.execution_engine import ExecutionEngine
from srtrsh_prompt_engine.resolver.service import ResolverService
from srtrsh_prompt_engine.spi.provider import (
    HydratedPrompt,
    ProviderExecutionRequest,
    ProviderSPI,
)
from srtrsh_prompt_engine.storage.repository import (
    EndpointCatalogRepository,
    OnlineTemplateRepository,
    PromptCaseRepository,
    VariableCatalogRepository,
)


# ============================================================================
# Resolver Router (Legacy / Data Plane direct resolution)
# ============================================================================


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

    return router


# ============================================================================
# Runtime API Router (C-23, 14-api-contracts/02-runtime-api.md)
# ============================================================================


def create_runtime_router(execution_engine: ExecutionEngine) -> APIRouter:
    router = APIRouter(prefix="/v1/runtime", tags=["Runtime API"])

    @router.post(
        "/execute",
        response_model=RuntimeOutcomeDTO,
        status_code=status.HTTP_200_OK,
    )
    async def execute_runtime_attempt(dto: RuntimeExecuteRequestDTO):
        """Execute one customer message attempt across one or more provider cycles."""
        outcome = await execution_engine.execute_attempt(
            conversation_id=dto.conversation_id,
            message=dto.message,
            runtime_values=dto.runtime_values,
            business_references=dto.business_references,
            case_key=dto.case_key,
        )

        options_dto = [
            InteractiveOptionDTO(label=opt.label, result=opt.result)
            for opt in outcome.options
        ]

        return RuntimeOutcomeDTO(
            executionId=outcome.execution_id,
            status=outcome.status.value,
            text=outcome.text,
            options=options_dto,
            sessionId=outcome.session_id,
            conversationId=outcome.conversation_id,
            sessionVersion=outcome.session_version,
            structuredState=outcome.structured_state,
            errorMessage=outcome.error_message,
            telemetry=outcome.telemetry,
        )

    @router.post(
        "/interaction",
        response_model=RuntimeOutcomeDTO,
        status_code=status.HTTP_200_OK,
    )
    async def submit_runtime_interaction(dto: RuntimeInteractionRequestDTO):
        """Submit customer button interaction result and continue execution cycle."""
        outcome = await execution_engine.execute_attempt(
            conversation_id=dto.conversation_id,
            interaction_result=dto.interaction_result,
            case_key=dto.case_key,
        )

        options_dto = [
            InteractiveOptionDTO(label=opt.label, result=opt.result)
            for opt in outcome.options
        ]

        return RuntimeOutcomeDTO(
            executionId=outcome.execution_id,
            status=outcome.status.value,
            text=outcome.text,
            options=options_dto,
            sessionId=outcome.session_id,
            conversationId=outcome.conversation_id,
            sessionVersion=outcome.session_version,
            structuredState=outcome.structured_state,
            errorMessage=outcome.error_message,
            telemetry=outcome.telemetry,
        )

    return router


# ============================================================================
# Manager API Router (C-21, C-24, 14-api-contracts/01-manager-api.md)
# ============================================================================


def create_manager_router(
    prompt_case_repo: PromptCaseRepository,
    endpoint_catalog_repo: EndpointCatalogRepository,
    variable_catalog_repo: VariableCatalogRepository,
) -> APIRouter:
    router = APIRouter(prefix="/v1/manager", tags=["Manager API"])

    # --- Prompt Cases ---
    @router.get("/prompt-cases", response_model=List[PromptCaseDTO])
    async def list_prompt_cases():
        cases = prompt_case_repo.list_all()
        result = []
        for c in cases:
            versions_dto = [
                PromptVersionDTO(
                    versionNumber=v.version_number,
                    state=v.state.value,
                    versionHash=v.version_hash,
                    createdAt=v.created_at.isoformat(),
                    publishedAt=v.published_at.isoformat() if v.published_at else None,
                    author=v.author,
                )
                for v in c.versions.values()
            ]
            result.append(
                PromptCaseDTO(
                    caseKey=c.case_key,
                    description=c.description,
                    activeVersionReference=c.active_version_reference,
                    publishedVersionNumber=c.published_version_number,
                    versions=versions_dto,
                )
            )
        return result

    @router.get("/prompt-cases/{case_key}", response_model=PromptCaseDTO)
    async def get_prompt_case(case_key: str):
        c = prompt_case_repo.get_by_key(case_key)
        if not c:
            raise HTTPException(status_code=404, detail=f"Prompt case '{case_key}' not found.")
        versions_dto = [
            PromptVersionDTO(
                versionNumber=v.version_number,
                state=v.state.value,
                versionHash=v.version_hash,
                createdAt=v.created_at.isoformat(),
                publishedAt=v.published_at.isoformat() if v.published_at else None,
                author=v.author,
            )
            for v in c.versions.values()
        ]
        return PromptCaseDTO(
            caseKey=c.case_key,
            description=c.description,
            activeVersionReference=c.active_version_reference,
            publishedVersionNumber=c.published_version_number,
            versions=versions_dto,
        )

    @router.post(
        "/prompt-cases/{case_key}/versions",
        response_model=PromptVersionDTO,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_prompt_version(case_key: str, dto: CreatePromptVersionDTO):
        case = prompt_case_repo.get_by_key(case_key)
        if not case:
            case = PromptCase(case_key=case_key)

        vars_list = [
            VariableDefinition(
                name=v.get("name", ""),
                type=v.get("type", "string"),
                required=v.get("required", True),
                default_value=v.get("defaultValue", None),
            )
            for v in dto.variables
        ]
        pm = None
        if dto.execution_configuration and "providerModel" in dto.execution_configuration:
            raw_pm = dto.execution_configuration["providerModel"]
            if raw_pm:
                pm = ProviderModel(
                    provider_id=raw_pm.get("providerId", "default"),
                    model_name=raw_pm.get("modelName", "default"),
                )

        defn = HashedVersionDefinition(
            template=Template(
                system_prompt=dto.template.get("systemPrompt", ""),
                user_prompt=dto.template.get("userPrompt", ""),
                format=dto.template.get("format", "STANDARD"),
            ),
            variables=vars_list,
            schema_contract=SchemaContract(
                schema_type=dto.schema_contract.get("schemaType", "JSON_SCHEMA") if dto.schema_contract else "JSON_SCHEMA",
                definition=dto.schema_contract.get("definition", {}) if dto.schema_contract else {},
            ),
            execution_configuration=ExecutionConfiguration(
                provider_model=pm,
                temperature=dto.execution_configuration.get("temperature") if dto.execution_configuration else None,
                timeout_duration=dto.execution_configuration.get("timeoutDuration") if dto.execution_configuration else None,
                max_output_tokens=dto.execution_configuration.get("maxOutputTokens") if dto.execution_configuration else None,
            ),
            override_policy=RuntimeOverridePolicy(
                allowed_parameters=set(dto.override_policy.get("allowedParameters", [])) if dto.override_policy else set(),
            ),
        )
        new_v = case.create_version(defn, author=dto.author)
        prompt_case_repo.save(case)

        return PromptVersionDTO(
            versionNumber=new_v.version_number,
            state=new_v.state.value,
            versionHash=new_v.version_hash,
            createdAt=new_v.created_at.isoformat(),
            publishedAt=None,
            author=new_v.author,
        )

    @router.post(
        "/prompt-cases/{case_key}/versions/{version_number}/publish",
        response_model=PromptVersionDTO,
        status_code=status.HTTP_200_OK,
    )
    async def publish_prompt_version(case_key: str, version_number: int):
        case = prompt_case_repo.get_by_key(case_key)
        if not case:
            raise HTTPException(status_code=404, detail=f"Prompt case '{case_key}' not found.")
        try:
            pub_v = case.publish_version(version_number)
            prompt_case_repo.save(case)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return PromptVersionDTO(
            versionNumber=pub_v.version_number,
            state=pub_v.state.value,
            versionHash=pub_v.version_hash,
            createdAt=pub_v.created_at.isoformat(),
            publishedAt=pub_v.published_at.isoformat() if pub_v.published_at else None,
            author=pub_v.author,
        )

    @router.post(
        "/prompt-cases/{case_key}/versions/{version_number}/deprecate",
        response_model=PromptVersionDTO,
        status_code=status.HTTP_200_OK,
    )
    async def deprecate_prompt_version(case_key: str, version_number: int):
        case = prompt_case_repo.get_by_key(case_key)
        if not case:
            raise HTTPException(status_code=404, detail=f"Prompt case '{case_key}' not found.")
        try:
            case.deprecate_version(version_number)
            prompt_case_repo.save(case)
            v = case.get_version(version_number)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return PromptVersionDTO(
            versionNumber=v.version_number,
            state=v.state.value,
            versionHash=v.version_hash,
            createdAt=v.created_at.isoformat(),
            publishedAt=v.published_at.isoformat() if v.published_at else None,
            author=v.author,
        )

    # --- Endpoint Catalog ---
    @router.get("/endpoint-catalog", response_model=List[EndpointCatalogEntryDTO])
    async def list_endpoint_catalog():
        entries = getattr(endpoint_catalog_repo, "list_endpoint_entries", endpoint_catalog_repo.list_entries)()
        return [
            EndpointCatalogEntryDTO(
                operationId=e.operation_id,
                description=e.description,
                humanDescription=e.human_description,
                inputVariables=e.input_variables,
                outputVariables=e.output_variables,
                isReadOnly=e.is_read_only,
                executionConstraints=e.execution_constraints,
                revision=e.revision,
            )
            for e in entries
        ]

    @router.post("/endpoint-catalog", status_code=status.HTTP_201_CREATED)
    async def save_endpoint_entry(dto: EndpointCatalogEntryDTO):
        try:
            entry = EndpointCatalogEntry(
                operation_id=dto.operation_id,
                description=dto.description,
                human_description=dto.human_description,
                input_variables=dto.input_variables,
                output_variables=dto.output_variables,
                is_read_only=dto.is_read_only,
                execution_constraints=dto.execution_constraints,
                revision=dto.revision,
            )
            getattr(endpoint_catalog_repo, "save_endpoint_entry", endpoint_catalog_repo.save_entry)(entry)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "created", "operationId": dto.operation_id}

    @router.delete("/endpoint-catalog/{operation_id}", status_code=status.HTTP_200_OK)
    async def delete_endpoint_entry(operation_id: str):
        getattr(endpoint_catalog_repo, "delete_endpoint_entry", endpoint_catalog_repo.delete_entry)(operation_id)
        return {"status": "deleted", "operationId": operation_id}

    # --- Variable Catalog ---
    @router.get("/variable-catalog", response_model=List[VariableCatalogEntryDTO])
    async def list_variable_catalog():
        entries = getattr(variable_catalog_repo, "list_variable_entries", variable_catalog_repo.list_entries)()
        return [
            VariableCatalogEntryDTO(
                variableId=v.variable_id,
                type=v.type.value,
                description=v.description,
                revision=v.revision,
            )
            for v in entries
        ]

    @router.post("/variable-catalog", status_code=status.HTTP_201_CREATED)
    async def save_variable_entry(dto: VariableCatalogEntryDTO):
        try:
            entry = VariableCatalogEntry(
                variable_id=dto.variable_id,
                type=VariableType(dto.type),
                description=dto.description,
                revision=dto.revision,
            )
            getattr(variable_catalog_repo, "save_variable_entry", variable_catalog_repo.save_entry)(entry)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "created", "variableId": dto.variable_id}

    @router.delete("/variable-catalog/{variable_id}", status_code=status.HTTP_200_OK)
    async def delete_variable_entry(variable_id: str):
        getattr(variable_catalog_repo, "delete_variable_entry", variable_catalog_repo.delete_entry)(variable_id)
        return {"status": "deleted", "variableId": variable_id}

    return router


# ============================================================================
# Localization API Router (C-27, 10-localization-and-templates/)
# ============================================================================


def create_localization_router(
    template_repo: OnlineTemplateRepository,
    provider: ProviderSPI,
) -> APIRouter:
    router = APIRouter(prefix="/v1/localization", tags=["Localization API"])

    @router.post(
        "/online/resolve",
        response_model=OnlineResolveResponseDTO,
        status_code=status.HTTP_200_OK,
    )
    async def resolve_online_template(dto: OnlineResolveRequestDTO):
        """Resolve online localized template from storage, or generate and persist if missing."""
        stored = template_repo.get_template(dto.case_key, dto.locale)
        if stored:
            try:
                rendered = stored.render(dto.variables)
                return OnlineResolveResponseDTO(
                    text=rendered,
                    locale=dto.locale,
                    caseKey=dto.case_key,
                    resolvedFrom="STORED_TEMPLATE",
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Generate on-the-fly using Provider SPI
        req = ProviderExecutionRequest(
            execution_id=f"loc_{uuid.uuid4().hex[:8]}",
            prompt_version_hash="00" * 32,
            provider="default",
            model="default",
            prompt=HydratedPrompt(
                system_prompt=f"Generate localized template for case {dto.case_key} in locale {dto.locale}",
                user_prompt=f"Variables: {list(dto.variables.keys())}",
            ),
        )
        resp = await provider.execute(req)
        gen_text = resp.content or resp.raw_text or ""
        if not gen_text:
            # Fallback to simple variable text
            gen_text = f"[{dto.case_key}:{dto.locale}] " + " ".join(f"${k}$" for k in dto.variables.keys())

        # Persist online template
        new_template = OnlineTemplate(
            case_key=dto.case_key,
            locale=dto.locale,
            template_text=gen_text,
            required_variables=list(dto.variables.keys()),
        )
        template_repo.save_template(new_template)
        rendered = new_template.render(dto.variables)

        return OnlineResolveResponseDTO(
            text=rendered,
            locale=dto.locale,
            caseKey=dto.case_key,
            resolvedFrom="GENERATED",
        )

    @router.post(
        "/offline/generate",
        response_model=OfflineGenerateResponseDTO,
        status_code=status.HTTP_200_OK,
    )
    async def generate_offline_resource(dto: OfflineGenerateRequestDTO):
        """Stateless generation function for Host Project offline resources (C-27).

        Prompt Engine does NOT persist the returned offline resource.
        """
        req = ProviderExecutionRequest(
            execution_id=f"offloc_{uuid.uuid4().hex[:8]}",
            prompt_version_hash="00" * 32,
            provider="default",
            model="default",
            prompt=HydratedPrompt(
                system_prompt=f"Translate text to {dto.target_locale} for offline UI resource {dto.resource_key}",
                user_prompt=dto.source_text,
            ),
        )
        resp = await provider.execute(req)
        gen_text = resp.content or resp.raw_text or dto.source_text

        # C-27: Prompt Engine MUST NOT persist the offline resource
        return OfflineGenerateResponseDTO(
            resourceKey=dto.resource_key,
            targetLocale=dto.target_locale,
            generatedText=gen_text,
        )

    return router
