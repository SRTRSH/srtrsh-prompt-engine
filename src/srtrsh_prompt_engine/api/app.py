"""FastAPI application factory for SRTRSH Prompt Engine."""

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from srtrsh_prompt_engine.api.routes import (
    create_localization_router,
    create_manager_router,
    create_resolver_router,
    create_runtime_router,
)
from srtrsh_prompt_engine.execution.dispatcher import OperationDispatcher
from srtrsh_prompt_engine.execution.execution_engine import ExecutionEngine
from srtrsh_prompt_engine.execution.session_engine import SessionEngine
from srtrsh_prompt_engine.resolver.service import ResolverService
from srtrsh_prompt_engine.seeds.system_bot import create_system_bot_master_intent_prompt
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.registry import ProviderRegistry
from srtrsh_prompt_engine.storage.repository import (
    InMemoryPromptRepository,
    PromptRepository,
    SQLitePromptEngineStorage,
)


def create_app(
    repository: Optional[PromptRepository] = None,
    provider_registry: Optional[ProviderRegistry] = None,
    storage: Optional[SQLitePromptEngineStorage] = None,
    execution_engine: Optional[ExecutionEngine] = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="SRTRSH Prompt Engine",
        description="Domain-agnostic prompt management, resolution, execution, and localization platform service.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    db_storage = storage or SQLitePromptEngineStorage(":memory:")
    repo = repository or db_storage

    # Seed built-in prompts if not already present
    if not repo.get_by_identity("system_bot_master_intent"):
        seed_prompt = create_system_bot_master_intent_prompt()
        repo.save(seed_prompt)

    registry = provider_registry or ProviderRegistry()
    default_provider = registry.get_or_default("default") or MockProvider()
    if not registry.get("mock"):
        registry.register("mock", default_provider)
    if not registry.get("default"):
        registry.register("default", default_provider)

    resolver_service = ResolverService(
        repository=repo, provider_registry=registry
    )

    # Wire ExecutionEngine for Runtime API
    if execution_engine:
        engine = execution_engine
    else:
        session_eng = SessionEngine(
            session_repository=db_storage,
            history_repository=db_storage,
        )
        dispatcher = OperationDispatcher(catalog_repository=db_storage)
        engine = ExecutionEngine(
            session_engine=session_eng,
            prompt_case_repo=db_storage,
            endpoint_catalog_repo=db_storage,
            variable_catalog_repo=db_storage,
            audit_repo=db_storage,
            dispatcher=dispatcher,
            provider=default_provider,
        )

    # Include all routers
    app.include_router(create_resolver_router(resolver_service))
    app.include_router(create_runtime_router(engine))
    app.include_router(
        create_manager_router(
            prompt_case_repo=db_storage,
            endpoint_catalog_repo=db_storage,
            variable_catalog_repo=db_storage,
        )
    )
    app.include_router(
        create_localization_router(
            template_repo=db_storage,
            provider=default_provider,
        )
    )

    # Store references in app.state
    app.state.repository = repo
    app.state.storage = db_storage
    app.state.provider_registry = registry
    app.state.resolver_service = resolver_service
    app.state.execution_engine = engine

    return app


app = create_app()
