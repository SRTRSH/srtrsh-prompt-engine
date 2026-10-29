# SRTRSH Prompt Engine

The Prompt Engine is a standalone, domain-agnostic platform service responsible for generic prompt lifecycle management, runtime prompt resolution, and LLM-assisted execution across SRTRSH communication channels.

## Architecture

Prompt Engine operates within the canonical closed-loop conversational pipeline:
```text
Person / Bot
    ↓
Communication Channel (e.g. Telegram)
    ↓
Channel Adapter (Thin Protocol Adapter)
    ↓
Communication Backend (Host Service)
    ↓
Prompt Engine (Domain-Agnostic LLM Transformation)
    ↓
operationId / outcome
    ↓
Communication Backend (executes operation against shared database)
    ↓
result
    ↓
Prompt Engine
    ↓
Communication Backend
    ↓
Channel Adapter
    ↓
Person
```

The Prompt Engine:
- Sits between Communication Backend (its sole Host Service) and LLM providers via Provider SPI.
- Authoritatively owns AI Sessions, conversation history, and execution context.
- Resolves immutable Prompt Versions identified by RFC 8785 (JCS) SHA-256 `PromptVersionHash`.
- Binds runtime variables and evaluates readiness before invoking LLM providers.
- Exposes Runtime API endpoints (`/v1/runtime/execute`, `/v1/runtime/interaction`) and Localization API endpoints.
- Returns structured outcomes (`NEEDS_USER_INPUT`, `CALL_OPERATION`, `COMPLETED`, `CONFIRMATION_REQUIRED`, `ERROR`) to Communication Backend.
- Remains strictly domain-agnostic, with no dependencies on client business entities, business rules, or direct database ownership of business backends. Channel Adapters and downstream business backends do not call Prompt Engine directly.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
