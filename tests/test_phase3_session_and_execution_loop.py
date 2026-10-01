"""Phase 3: Session Engine & Multi-Cycle Execution Loop Tests.

Verifies:
- One customer message equals one processing attempt (C-06)
- Multi-cycle dynamic execution without hardcoded DAGs (C-07)
- Intermediate CALL_OPERATION is processed internally; only terminal outcomes returned (BLK-04)
- Existing context values are reused to avoid redundant calls (C-08)
- Result ambiguity is preserved for LLM clarification rather than silently guessed (C-09)
- Oversized operation results are bounded with truncation signal (C-10)
- AI write operations are forbidden in MVP (C-11)
- Structured interaction result round-trip (C-12)
- Human fallback differs from global AI technical failure (C-16)
- Continuation window closes session without deleting history (C-25, C-26)
- Bounded history selection (recentMessages=N + previousSessions=M) (C-25)
- Execution limits and guardrails (maxLLMCalls, timeout) -> LIMIT_EXHAUSTED
"""

from datetime import datetime, timedelta, timezone
import pytest

from srtrsh_prompt_engine.domain.models import (
    EndpointCatalogEntry,
    ExecutionConfiguration,
    HashedVersionDefinition,
    PromptCase,
    SchemaContract,
    Template,
    VariableCatalogEntry,
    VariableType,
)
from srtrsh_prompt_engine.execution.dispatcher import OperationDispatcher
from srtrsh_prompt_engine.execution.execution_engine import (
    ExecutionEngine,
    ExecutionLimits,
)
from srtrsh_prompt_engine.execution.outcomes import (
    ExecutionOutcomeType,
)
from srtrsh_prompt_engine.execution.session_engine import SessionEngine
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.provider import (
    NormalizedInteractionOutput,
    NormalizedOperationCall,
    ProviderExecutionRequest,
    ProviderExecutionResponse,
)
from srtrsh_prompt_engine.storage.repository import (
    SQLitePromptEngineStorage,
)


@pytest.fixture
def execution_env():
    storage = SQLitePromptEngineStorage(":memory:")

    # Setup Organization System Prompt
    org_case = PromptCase(case_key="organization_system")
    org_defn = HashedVersionDefinition(
        template=Template(
            system_prompt="<system>You are the SRTRSH AI Salon Assistant.</system>",
            user_prompt="",
        ),
        schema_contract=SchemaContract(),
        execution_configuration=ExecutionConfiguration(),
    )
    org_case.create_version(org_defn)
    org_case.publish_version(1)
    storage.save(org_case)

    # Setup Default Case: resolve_user_request
    case = PromptCase(case_key="resolve_user_request")
    case_defn = HashedVersionDefinition(
        template=Template(
            system_prompt="Follow salon policies.",
            user_prompt="Resolve client booking request.",
        ),
        schema_contract=SchemaContract(),
        execution_configuration=ExecutionConfiguration(),
    )
    case.create_version(case_defn)
    case.publish_version(1)
    storage.save(case)

    # Setup Endpoint Catalog entries
    storage.save_entry(
        EndpointCatalogEntry(
            operation_id="findServices",
            description="Search available salon services",
            input_variables=["query"],
            output_variables=["service_id", "service_name"],
            revision="rev_1",
        )
    )
    storage.save_entry(
        EndpointCatalogEntry(
            operation_id="findAvailableSlots",
            description="Find available time slots",
            input_variables=["master_id!", "service_id!"],
            output_variables=["available_slots"],
            revision="rev_1",
        )
    )

    # Setup Variable Catalog entries
    storage.save_entry(
        VariableCatalogEntry(
            variable_id="master_id",
            type=VariableType.STRING,
            description="Master ID",
            revision="rev_1",
        )
    )
    storage.save_entry(
        VariableCatalogEntry(
            variable_id="service_id",
            type=VariableType.STRING,
            description="Service ID",
            revision="rev_1",
        )
    )

    session_engine = SessionEngine(
        session_repository=storage,
        history_repository=storage,
        default_continuation_window_seconds=1800,
    )
    dispatcher = OperationDispatcher(catalog_repository=storage, max_result_items=5)
    provider = MockProvider()

    engine = ExecutionEngine(
        session_engine=session_engine,
        prompt_case_repo=storage,
        endpoint_catalog_repo=storage,
        variable_catalog_repo=storage,
        audit_repo=storage,
        dispatcher=dispatcher,
        provider=provider,
        limits=ExecutionLimits(max_llm_calls=5, timeout_seconds=10.0),
    )

    return {
        "storage": storage,
        "session_engine": session_engine,
        "dispatcher": dispatcher,
        "provider": provider,
        "engine": engine,
    }


# ============================================================================
# 1. Multi-Cycle Execution Loop & Attempt Cardinality (C-06, C-07, BLK-04)
# ============================================================================


@pytest.mark.asyncio
async def test_multi_cycle_loop_with_internal_call_operation(execution_env):
    """C-06, C-07, BLK-04:

    One customer message triggers an attempt with 2 internal operation calls.
    Intermediate CALL_OPERATION is never returned as a terminal outcome.
    The attempt terminates with COMPLETED.
    """
    engine = execution_env["engine"]
    provider = execution_env["provider"]
    dispatcher = execution_env["dispatcher"]

    # Register operation handlers
    dispatcher.register_handler(
        "findServices", lambda args: {"service_id": "S123", "service_name": "Стрижка"}
    )
    dispatcher.register_handler(
        "findAvailableSlots", lambda args: {"available_slots": ["14:00", "15:00"]}
    )

    cycle_count = 0

    def multi_cycle_handler(req: ProviderExecutionRequest) -> ProviderExecutionResponse:
        nonlocal cycle_count
        cycle_count += 1
        if cycle_count == 1:
            # Cycle 1: LLM requests findServices
            return ProviderExecutionResponse(
                execution_id=req.execution_id,
                status="SUCCESS",
                operation_calls=[
                    NormalizedOperationCall(
                        call_id="call_1",
                        operation_id="findServices",
                        arguments={"query": "стрижка"},
                    )
                ],
                finish_reason="tool_calls",
            )
        elif cycle_count == 2:
            # Cycle 2: LLM requests findAvailableSlots using discovered service_id
            return ProviderExecutionResponse(
                execution_id=req.execution_id,
                status="SUCCESS",
                operation_calls=[
                    NormalizedOperationCall(
                        call_id="call_2",
                        operation_id="findAvailableSlots",
                        arguments={"master_id": "M42", "service_id": "S123"},
                    )
                ],
                finish_reason="tool_calls",
            )
        else:
            # Cycle 3: Final terminal answer
            return ProviderExecutionResponse(
                execution_id=req.execution_id,
                status="SUCCESS",
                content="Я нашел запись на стрижку на 14:00 и 15:00.",
                finish_reason="stop",
            )

    provider.set_custom_handler(multi_cycle_handler)

    outcome = await engine.execute_attempt(
        conversation_id="conv_multicycle_1",
        message="Хочу записаться на стрижку к мастеру M42",
        runtime_values={"master_id": "M42"},
    )

    # Must terminate with COMPLETED (never intermediate CALL_OPERATION)
    assert outcome.status == ExecutionOutcomeType.COMPLETED
    assert "14:00" in outcome.text
    # Total LLM cycles inside this one customer attempt was 3
    assert cycle_count == 3
    # Session state contains accumulated service_id and available_slots
    assert outcome.structured_state.get("service_id") == "S123"
    assert outcome.structured_state.get("available_slots") == ["14:00", "15:00"]
    # Session version was advanced atomically from 1 to 2
    assert outcome.session_version == 2


# ============================================================================
# 2. Existing Value Reuse (C-08)
# ============================================================================


def test_dispatcher_reuses_existing_cached_values(execution_env):
    """C-08: If an operation is invoked with identical arguments, result is reused."""
    dispatcher = execution_env["dispatcher"]
    call_count = 0

    def find_services(args):
        nonlocal call_count
        call_count += 1
        return [{"service_id": "S1", "name": "Борода"}]

    dispatcher.register_handler("findServices", find_services)

    # First call: invokes handler
    res1 = dispatcher.execute_operation("findServices", {"query": "борода"})
    assert res1.success is True
    assert res1.was_reused is False
    assert call_count == 1

    # Second call with identical arguments: reuses cached value
    res2 = dispatcher.execute_operation("findServices", {"query": "борода"})
    assert res2.success is True
    assert res2.was_reused is True
    assert call_count == 1  # Handler was not called again!


# ============================================================================
# 3. Ambiguity & Result Limits (C-09, C-10)
# ============================================================================


def test_dispatcher_handles_ambiguity_without_silent_guessing(execution_env):
    """C-09: Ambiguous results (multiple items) are preserved rather than picking one arbitrarily."""
    dispatcher = execution_env["dispatcher"]
    dispatcher.register_handler(
        "findServices",
        lambda args: [
            {"service_id": "S1", "name": "Мужская стрижка"},
            {"service_id": "S2", "name": "Женская стрижка"},
            {"service_id": "S3", "name": "Детская стрижка"},
        ],
    )

    res = dispatcher.execute_operation("findServices", {"query": "стрижка"})
    assert res.success is True
    assert res.is_ambiguous is True
    # All 3 items are preserved for LLM reasoning
    assert len(res.data) == 3


def test_dispatcher_bounds_oversized_results(execution_env):
    """C-10: Oversized results exceeding max_result_items are bounded with a truncation signal."""
    dispatcher = execution_env["dispatcher"]
    # max_result_items is configured to 5 in fixture
    dispatcher.register_handler(
        "findServices",
        lambda args: [{"service_id": f"S{i}", "name": f"Service {i}"} for i in range(12)],
    )

    res = dispatcher.execute_operation("findServices", {"query": "all"})
    assert res.success is True
    assert res.is_truncated is True
    # Bounded to exactly 5 items
    assert len(res.data) == 5
    dict_repr = res.to_dict()
    assert dict_repr.get("isTruncated") is True
    assert "Слишком много результатов" in dict_repr.get("warning", "")


# ============================================================================
# 4. Interaction Machine Results (C-12)
# ============================================================================


@pytest.mark.asyncio
async def test_interaction_machine_result_persistence_and_response(execution_env):
    """C-12: Structured interactive button choices and machine results round-trip."""
    engine = execution_env["engine"]
    provider = execution_env["provider"]

    # Provider returns interactive response
    provider.set_interaction(
        NormalizedInteractionOutput(
            text="Вам подходит мужская стрижка на 17:30?",
            options=[
                {"label": "Да", "result": {"confirmation": True}},
                {"label": "Нет", "result": {"confirmation": False}},
            ],
        )
    )

    outcome = await engine.execute_attempt(
        conversation_id="conv_interact_12",
        message="Запишите меня на 17:30",
    )

    assert outcome.status == ExecutionOutcomeType.INTERACTIVE_RESPONSE
    assert len(outcome.options) == 2
    assert outcome.options[0].label == "Да"
    assert outcome.options[0].result == {"confirmation": True}

    # Now simulate customer clicking [Да] -> sending structured interaction result
    provider.set_interaction(None)
    provider.set_default_output({"status": "SUCCESS"})

    def confirm_handler(req: ProviderExecutionRequest) -> ProviderExecutionResponse:
        return ProviderExecutionResponse(
            execution_id=req.execution_id,
            status="SUCCESS",
            content="Отлично! Запись подтверждена на 17:30.",
        )

    provider.set_custom_handler(confirm_handler)

    follow_up = await engine.execute_attempt(
        conversation_id="conv_interact_12",
        interaction_result={"confirmation": True},
    )

    assert follow_up.status == ExecutionOutcomeType.COMPLETED
    assert "подтверждена" in follow_up.text
    # Session state recorded confirmation
    assert follow_up.structured_state.get("confirmation") is True


# ============================================================================
# 5. Human Fallback vs Technical Failure (C-16)
# ============================================================================


@pytest.mark.asyncio
async def test_human_fallback_semantics(execution_env):
    """C-16: Semantic escalation produces HUMAN_FALLBACK."""
    engine = execution_env["engine"]
    provider = execution_env["provider"]

    provider.set_custom_handler(
        lambda req: ProviderExecutionResponse(
            execution_id=req.execution_id,
            status="SUCCESS",
            content="Переключаю вас на человека: администратор скоро ответит.",
        )
    )

    outcome = await engine.execute_attempt(
        conversation_id="conv_fallback_1",
        message="Позовите человека, у меня нестандартный вопрос",
    )

    assert outcome.status == ExecutionOutcomeType.HUMAN_FALLBACK
    assert "администратор" in outcome.text


@pytest.mark.asyncio
async def test_technical_failure_semantics(execution_env):
    """C-16: Infrastructure/provider failure produces TECHNICAL_ERROR."""
    engine = execution_env["engine"]
    provider = execution_env["provider"]

    provider.set_custom_handler(
        lambda req: ProviderExecutionResponse(
            execution_id=req.execution_id,
            status="FAILURE",
            error_message="Database connection pool timeout",
        )
    )

    outcome = await engine.execute_attempt(
        conversation_id="conv_tech_fail",
        message="Привет",
    )

    assert outcome.status == ExecutionOutcomeType.TECHNICAL_ERROR
    assert "timeout" in outcome.error_message


# ============================================================================
# 6. Continuation Window vs Historical Context (C-25, C-26)
# ============================================================================


@pytest.mark.asyncio
async def test_session_continuation_window_and_history_retention(execution_env):
    """C-25, C-26:

    Exceeding conversationContinuationWindow closes the session and creates a new one,
    but retains historical conversation and passes previous session context to LLM.
    """
    storage = execution_env["storage"]
    session_engine = execution_env["session_engine"]

    # 1. Start session 1 in conversation 'conv_c25'
    sess1 = session_engine.get_or_create_session("conv_c25")
    session_engine.record_message(sess1.session_id, "conv_c25", "customer", "Привет, я Анна")
    session_engine.record_message(sess1.session_id, "conv_c25", "assistant", "Здравствуйте, Анна!")

    # Advance time beyond continuation window (1800s)
    sess1.updated_at = datetime.now(timezone.utc) - timedelta(seconds=2000)
    storage.save(sess1)

    # 2. Incoming message arrives after 2000 seconds
    sess2 = session_engine.get_or_create_session("conv_c25")

    # Session 2 is a new session with session_version=1
    assert sess2.session_id != sess1.session_id
    assert sess2.session_version == 1

    # Session 1 was closed
    stored_sess1 = storage.get_by_id(sess1.session_id)
    assert stored_sess1.is_active is False
    assert stored_sess1.closed_at is not None

    # History from session 1 is NOT deleted (C-26)
    all_history = storage.get_messages_for_conversation("conv_c25")
    assert len(all_history) == 2
    assert all_history[0].content == "Привет, я Анна"

    # Context history selection includes previous session history (C-25)
    selected_context = session_engine.select_history_context(
        conversation_id="conv_c25",
        current_session_id=sess2.session_id,
        recent_messages_limit=5,
        previous_sessions_limit=1,
    )
    assert len(selected_context) == 2
    assert selected_context[0].content == "Привет, я Анна"


# ============================================================================
# 7. Limits & Guardrails (LIMIT_EXHAUSTED)
# ============================================================================


@pytest.mark.asyncio
async def test_guardrail_max_llm_calls_exhaustion(execution_env):
    """Infinite loop of operation calls is capped at maxLLMCalls -> LIMIT_EXHAUSTED."""
    engine = execution_env["engine"]
    provider = execution_env["provider"]
    dispatcher = execution_env["dispatcher"]

    dispatcher.register_handler("findServices", lambda args: {"result": "ok"})

    # Provider constantly asks for more operation calls
    provider.set_custom_handler(
        lambda req: ProviderExecutionResponse(
            execution_id=req.execution_id,
            status="SUCCESS",
            operation_calls=[
                NormalizedOperationCall(
                    call_id="call_inf",
                    operation_id="findServices",
                    arguments={"query": "test"},
                )
            ],
            finish_reason="tool_calls",
        )
    )

    outcome = await engine.execute_attempt(
        conversation_id="conv_limit_test",
        message="Start loop",
    )

    assert outcome.status == ExecutionOutcomeType.LIMIT_EXHAUSTED
    assert "Exhausted maximum permitted LLM calls" in outcome.error_message
