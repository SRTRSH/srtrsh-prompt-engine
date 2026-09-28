# SRTRSH Prompt Engine

The Prompt Engine is a standalone, domain-agnostic platform service responsible for generic prompt lifecycle management, runtime prompt resolution, and LLM-assisted execution across SRTRSH communication channels.

## Architecture

The canonical pipeline is:
```text
Person → Bot → Communication Channel → Channel Adapter → Prompt Engine → Appropriate Backend
```

The Prompt Engine:
- Accepts `promptIdentity`, `rawInput`, and `runtimeContext`.
- Resolves immutable Prompt Versions identified by RFC 8785 (JCS) SHA-256 `PromptVersionHash`.
- Binds runtime variables and evaluates readiness:
  - If required variables are missing: returns `isReady = false` with `missingVariables = [...]` without invoking LLM providers.
  - If all required variables are resolved: returns `isReady = true`, invokes the configured provider via the Provider SPI, and validates output against `SchemaContract` (JSON Schema).
- Exposes transport-neutral Resolver API (`POST /v1/resolver/resolve-and-execute`, `POST /v1/resolver/resolve`, `POST /v1/resolver/execute`).
- Remains strictly domain-agnostic, with no dependencies on client business entities, business rules, or direct database ownership of business backends.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
