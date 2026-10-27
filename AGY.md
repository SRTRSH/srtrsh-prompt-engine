# AGY — SRTRSH Prompt Engine

## Purpose and entry path

These are repository-specific instructions for `SRTRSH/srtrsh-prompt-engine`. The global workflow and precedence rules are defined in `SRTRSH/srtrsh-docs/AGY.md`.

Before any audit or implementation:
1. Read the global `srtrsh-docs/AGY.md`.
2. Read `srtrsh-docs/WikiDocs/14-prompt-engine/README.md` and follow its prescribed reading order for the affected behavior.
3. Read relevant normative contracts, invariants, scenarios, conformance matrix, and Definition of Done.
4. Inspect this repository's `README.md`, `pyproject.toml`, source, tests, and current implementation.
5. Inspect `srtrsh-communication-backend` for the host integration contract when relevant, and inspect `srtrsh-infrastructure` for shared schema/migration facts when persistence is involved.

Do not ask the human to restate information already available in those sources.

## Component responsibility and boundaries

Prompt Engine is a standalone, domain-agnostic platform service. It owns Prompt Cases and immutable Prompt Versions, prompt resolution/assembly, variable readiness, AI Sessions, conversation history and execution context, controlled AI execution through Provider SPI, structured interactions, localization/template behavior, limits, fallback semantics, auditability and reproducibility as defined by the canonical specification.

It MUST NOT:
- own or mutate business-domain truth;
- directly create/change Orders, Appointments, or other business entities;
- implement business authorization or business mutations belonging to Communication Backend;
- call Master Backend, Admin Backend, Payment Backend, or other business backends directly;
- execute arbitrary URLs or arbitrary operations supplied by an LLM;
- let untrusted prompt/user content alter catalogs, system configuration, or authorization;
- treat cache as authoritative or make correctness depend on cache availability;
- bypass Provider SPI with provider-specific logic spread through the domain/runtime code.

Communication Backend is the Host Service. Prompt Engine returns structured outcomes/operation identifiers according to the contract; Communication Backend authorizes and executes permitted business operations.

## Technology stack and host constraints

This repository currently uses Python >=3.12, FastAPI, Pydantic, pytest/pytest-asyncio and a `src/` package layout, as declared in `pyproject.toml`. Preserve these conventions unless an approved specification or explicit human instruction requires a change.

Do not impose Java/Spring, a new transport, ORM, migration framework, database product, deployment platform, authentication mechanism, or observability stack based on examples in the Prompt Engine specification. Host-owned choices must follow the actual SRTRSH host/project configuration and approved contracts.

Provider SDKs must remain behind Provider SPI. Add dependencies only when needed and compatible with the established project.

## API and persistence contracts

- Implement Prompt Engine-specific semantics separately from host-owned transport, authentication, routing, common headers and common error-envelope conventions.
- Preserve stable identifiers, Prompt Version immutability and JCS-compatible canonicalization/SHA-256 identity requirements.
- Enforce resource validation, lifecycle rules, compatibility, concurrency/optimistic concurrency, bounded collection retrieval/pagination, idempotency and error semantics where required by the applicable resource contract.
- Do not add pagination or idempotency as boilerplate where the operation semantics do not require them; do not omit them where unbounded results, retries, mutation, or concurrency make them required.
- Use the existing storage/repository boundary and canonical common-database ownership. Coordinate physical DDL changes through `srtrsh-infrastructure`; do not add a separate business database or duplicate schema migrations here.
- Preserve transaction and persistence guarantees. Treat AI output, user messages, conversation history, and retrieved content as untrusted data.

## Implementation workflow

For each requested capability:
1. Map it to normative specification documents and conformance scenarios.
2. Inspect existing code/tests to classify behavior as implemented, partial, missing, or conflicting.
3. Implement only the required gap, retaining valid existing behavior.
4. Add tests for the requirement and its relevant failure, boundary, persistence, concurrency, and security cases.
5. Update existing tests where the canonical specification intentionally supersedes obsolete behavior.
6. Run the relevant tests and the full `pytest` suite when practical.
7. Report requirement-to-code/test evidence, tests actually run, and unresolved decisions.

The implementation is not complete merely because an endpoint exists or unit tests pass. The Prompt Engine Definition of Done and conformance matrix are the acceptance criteria. Do not replace required dynamic execution semantics with an unapproved mandatory DAG/scheduler model.

## Git

Follow the global Git policy. Do not commit, push, merge, or open a pull request unless the human explicitly requests it. A task branch is not permission to perform those operations.
