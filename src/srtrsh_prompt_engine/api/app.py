"""FastAPI application factory for SRTRSH Prompt Engine."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from srtrsh_prompt_engine.api.routes import create_resolver_router
from srtrsh_prompt_engine.resolver.service import ResolverService
from srtrsh_prompt_engine.seeds.system_bot import create_system_bot_master_intent_prompt
from srtrsh_prompt_engine.spi.registry import ProviderRegistry
from srtrsh_prompt_engine.storage.repository import (
    InMemoryPromptRepository,
    PromptRepository,
)


def create_app(
    repository: PromptRepository | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="SRTRSH Prompt Engine",
        description="Domain-agnostic prompt management, resolution, and execution platform service.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    repo = repository or InMemoryPromptRepository()
    # Seed built-in prompts if not already present
    if not repo.get_by_identity("system_bot_master_intent"):
        seed_prompt = create_system_bot_master_intent_prompt()
        repo.save(seed_prompt)

    registry = provider_registry or ProviderRegistry()
    resolver_service = ResolverService(
        repository=repo, provider_registry=registry
    )

    router = create_resolver_router(resolver_service)
    app.include_router(router)

    # Store references in app.state for access if needed
    app.state.repository = repo
    app.state.provider_registry = registry
    app.state.resolver_service = resolver_service

    return app


app = create_app()
