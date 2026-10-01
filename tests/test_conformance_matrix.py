"""Complete Conformance Verification Suite for C-01 through C-31.

Authoritative reference:
WikiDocs/14-prompt-engine/19-scenarios-and-conformance/10-conformance-matrix.md
"""

from datetime import datetime, timedelta, timezone
import pytest

from srtrsh_prompt_engine.compiler.catalog_compiler import (
    CatalogCompiler,
    CompiledCatalogCache,
)
from srtrsh_prompt_engine.domain.models import (
    AISession,
    AuditRecord,
    EndpointCatalogEntry,
    ExecutionConfiguration,
    HashedVersionDefinition,
    OnlineTemplate,
    PromptCase,
    RuntimeOverridePolicy,
    SchemaContract,
    SessionConcurrencyError,
    SessionMessage,
    Template,
    VariableCatalogEntry,
    VariableDefinition,
    VariableType,
    VersionState,
)
from srtrsh_prompt_engine.execution.context_assembler import ContextAssembler
from srtrsh_prompt_engine.execution.dispatcher import OperationDispatcher
from srtrsh_prompt_engine.execution.execution_engine import (
    ExecutionEngine,
    ExecutionLimits,
)
from srtrsh_prompt_engine.execution.outcomes import ExecutionOutcomeType
from srtrsh_prompt_engine.execution.session_engine import SessionEngine
from srtrsh_prompt_engine.spi.generic_llm import normalize_http_error
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.provider import (
    HydratedPrompt,
    NormalizedInteractionOutput,
    NormalizedOperationCall,
    ProviderErrorCategory,
    ProviderExecutionRequest,
    ProviderExecutionResponse,
)
from srtrsh_prompt_engine.storage.repository import (
    InMemoryPromptCaseRepository,
    SQLitePromptEngineStorage,
)


def _sample_defn(text: str = "Test prompt"):
    return HashedVersionDefinition(
        template=Template(system_prompt=text, user_prompt="User: {{query}}"),
        variables=[VariableDefinition(name="query", type="string", required=True)],
        schema_contract=SchemaContract(),
        execution_configuration=ExecutionConfiguration(temperature=0.0),
    )


# ----------------------------------------------------------------------------
# C-01: Business authority remains in Client Backend
# ----------------------------------------------------------------------------
def test_c01_business_authority_remains_in_backend():
    session = AISession(
        conversation_id="conv_c01",
        business_references={"customer_id": "C777", "master_id": "M42"},
    )
    assert session.business_references["customer_id"] == "C777"
    assert "orders" not in AISession.model_fields
    assert "appointments" not in AISession.model_fields


# ----------------------------------------------------------------------------
# C-02: Published Prompt Versions are immutable
# ----------------------------------------------------------------------------
def test_c02_published_versions_immutable():
    case = PromptCase(case_key="booking_flow")
    v1 = case.create_version(_sample_defn("Prompt v1"))
    case.publish_version(1)
    assert v1.state == VersionState.PUBLISHED

    with pytest.raises(ValueError, match="Published versions are immutable"):
        v1.definition = _sample_defn("Hacked prompt")


# ----------------------------------------------------------------------------
# C-03: Version number and Prompt Version Hash are distinct
# ----------------------------------------------------------------------------
def test_c03_version_number_and_hash_distinct():
    case = PromptCase(case_key="test_case")
    v1 = case.create_version(_sample_defn("Identities test"))
    case.publish_version(1)

    assert v1.version_number == 1
    assert isinstance(v1.version_hash, str)
    assert len(v1.version_hash) == 64
    assert v1.version_hash != str(v1.version_number)


# ----------------------------------------------------------------------------
# C-04: Endpoint IDs are stable and arbitrary URLs are forbidden
# ----------------------------------------------------------------------------
def test_c04_endpoint_ids_stable_arbitrary_urls_forbidden():
    storage = SQLitePromptEngineStorage(":memory:")
    storage.save_endpoint_entry(
        EndpointCatalogEntry(
            operation_id="findServices",
            description="Find services",
            input_variables=["query"],
            output_variables=["service_id"],
        )
    )
    dispatcher = OperationDispatcher(catalog_repository=storage)

    # Valid stable operationId succeeds
    res_ok = dispatcher.execute_operation("findServices", {"query": "haircut"})
    assert res_ok.success is True

    # Arbitrary URL is forbidden and fails validation
    res_url = dispatcher.execute_operation("https://api.example.com/deleteOrder", {})
    assert res_url.success is False
    assert "not registered" in res_url.error_message


# ----------------------------------------------------------------------------
# C-05: Variable contracts are stable and compatibility checked
# ----------------------------------------------------------------------------
def test_c05_variable_contracts_stable_and_compatibility_checked():
    # Valid variable types
    for vt in ["string", "integer", "number", "boolean", "datetime", "array", "object"]:
        entry = VariableCatalogEntry(variable_id=f"var_{vt}", type=VariableType(vt), description="desc")
        assert entry.type == VariableType(vt)

    # Incompatible unknown type rejected
    with pytest.raises(ValueError, match="Invalid variable type"):
        VariableCatalogEntry(variable_id="bad_var", type="unknown_type", description="desc")  # type: ignore


# ----------------------------------------------------------------------------
# C-06: One customer message equals one processing attempt
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_c06_one_customer_message_equals_one_attempt():
    storage = SQLitePromptEngineStorage(":memory:")
    provider = MockProvider()
    cycles = 0

    def handler(req):
        nonlocal cycles
        cycles += 1
        if cycles < 3:
            return ProviderExecutionResponse(
                execution_id=req.execution_id,
                status="SUCCESS",
                operation_calls=[
                    NormalizedOperationCall(call_id=f"c_{cycles}", operation_id="findServices", arguments={})
                ],
                finish_reason="tool_calls",
            )
        return ProviderExecutionResponse(
            execution_id=req.execution_id, status="SUCCESS", content="Final answer"
        )

    provider.set_custom_handler(handler)
    storage.save_endpoint_entry(EndpointCatalogEntry(operation_id="findServices", description="desc"))

    session_engine = SessionEngine(storage, storage)
    engine = ExecutionEngine(
        session_engine=session_engine,
        prompt_case_repo=storage,
        endpoint_catalog_repo=storage,
        variable_catalog_repo=storage,
        audit_repo=storage,
        dispatcher=OperationDispatcher(storage),
        provider=provider,
    )

    # Exactly 1 customer message sent
    outcome = await engine.execute_attempt(conversation_id="conv_c06", message="First message")
    # Inside this one attempt, multiple provider cycles executed
    assert cycles == 3
    assert outcome.status == ExecutionOutcomeType.COMPLETED


# ----------------------------------------------------------------------------
# C-07: Dynamic execution does not require DAG/stepId/dependsOn
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_c07_dynamic_execution_without_dag():
    storage = SQLitePromptEngineStorage(":memory:")
    storage.save_endpoint_entry(EndpointCatalogEntry(operation_id="findServices", description="desc"))
    provider = MockProvider()
    provider.set_default_output({"service_id": "S1"})

    session_engine = SessionEngine(storage, storage)
    engine = ExecutionEngine(
        session_engine=session_engine,
        prompt_case_repo=storage,
        endpoint_catalog_repo=storage,
        variable_catalog_repo=storage,
        audit_repo=storage,
        dispatcher=OperationDispatcher(storage),
        provider=provider,
    )

    # Executes dynamically without requiring stepId or dependsOn DAG metadata
    outcome = await engine.execute_attempt(conversation_id="conv_c07", message="Book service")
    assert outcome.status in (ExecutionOutcomeType.COMPLETED, ExecutionOutcomeType.NEEDS_USER_INPUT)


# ----------------------------------------------------------------------------
# C-08: Existing context values are reused
# ----------------------------------------------------------------------------
def test_c08_existing_context_values_reused():
    storage = SQLitePromptEngineStorage(":memory:")
    storage.save_endpoint_entry(EndpointCatalogEntry(operation_id="findServices", description="desc"))
    dispatcher = OperationDispatcher(storage)

    invocations = 0

    def service_handler(args):
        nonlocal invocations
        invocations += 1
        return [{"id": "S10"}]

    dispatcher.register_handler("findServices", service_handler)

    r1 = dispatcher.execute_operation("findServices", {"query": "nails"})
    assert r1.was_reused is False
    assert invocations == 1

    r2 = dispatcher.execute_operation("findServices", {"query": "nails"})
    assert r2.was_reused is True
    assert invocations == 1  # Reused from context cache!


# ----------------------------------------------------------------------------
# C-09: Ambiguous operation results are not silently guessed
# ----------------------------------------------------------------------------
def test_c09_ambiguous_results_not_silently_guessed():
    storage = SQLitePromptEngineStorage(":memory:")
    storage.save_endpoint_entry(EndpointCatalogEntry(operation_id="findServices", description="desc"))
    dispatcher = OperationDispatcher(storage)
    dispatcher.register_handler("findServices", lambda args: [{"id": 1}, {"id": 2}])

    res = dispatcher.execute_operation("findServices", {"query": "cut"})
    assert res.is_ambiguous is True
    assert len(res.data) == 2  # Preserves all options


# ----------------------------------------------------------------------------
# C-10: Oversized operation results are bounded
# ----------------------------------------------------------------------------
def test_c10_oversized_results_bounded():
    storage = SQLitePromptEngineStorage(":memory:")
    storage.save_endpoint_entry(EndpointCatalogEntry(operation_id="findServices", description="desc"))
    dispatcher = OperationDispatcher(storage, max_result_items=3)
    dispatcher.register_handler("findServices", lambda args: [{"id": i} for i in range(10)])

    res = dispatcher.execute_operation("findServices", {})
    assert res.is_truncated is True
    assert len(res.data) == 3


# ----------------------------------------------------------------------------
# C-11: AI write operations are unavailable in MVP
# ----------------------------------------------------------------------------
def test_c11_ai_write_operations_forbidden_in_mvp():
    with pytest.raises(ValueError, match="Write operation 'deleteCustomer' is not permitted"):
        EndpointCatalogEntry(
            operation_id="deleteCustomer",
            description="Delete customer record",
            is_read_only=False,
        )


# ----------------------------------------------------------------------------
# C-12: Interaction machine results are persisted
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_c12_interaction_machine_results_persisted():
    storage = SQLitePromptEngineStorage(":memory:")
    session_engine = SessionEngine(storage, storage)
    engine = ExecutionEngine(
        session_engine=session_engine,
        prompt_case_repo=storage,
        endpoint_catalog_repo=storage,
        variable_catalog_repo=storage,
        audit_repo=storage,
        dispatcher=OperationDispatcher(storage),
        provider=MockProvider(),
    )

    outcome = await engine.execute_attempt(
        conversation_id="conv_c12",
        interaction_result={"confirmation": True, "selected_slot": "16:00"},
    )
    assert outcome.structured_state.get("confirmation") is True
    assert outcome.structured_state.get("selected_slot") == "16:00"


# ----------------------------------------------------------------------------
# C-13: Templates are deterministically validated
# ----------------------------------------------------------------------------
def test_c13_templates_deterministically_validated():
    tmpl = OnlineTemplate(
        case_key="reminder",
        locale="ru",
        template_text="Hello $name$, appointment at $time$",
        required_variables=["name", "time"],
    )
    # Valid
    valid, missing = tmpl.validate_variables({"name": "Anna", "time": "12:00"})
    assert valid is True
    assert tmpl.render({"name": "Anna", "time": "12:00"}) == "Hello Anna, appointment at 12:00"

    # Missing
    valid_bad, missing_bad = tmpl.validate_variables({"name": "Anna"})
    assert valid_bad is False
    assert "time" in missing_bad


# ----------------------------------------------------------------------------
# C-14: Provider SDKs remain behind Provider SPI
# ----------------------------------------------------------------------------
def test_c14_provider_sdks_behind_spi():
    provider = MockProvider()
    assert hasattr(provider, "execute")
    assert hasattr(provider, "get_capabilities")
    # Domain models have zero dependency on OpenAI / Anthropic SDK types
    import srtrsh_prompt_engine.domain.models as dm
    assert not hasattr(dm, "openai")
    assert not hasattr(dm, "anthropic")


# ----------------------------------------------------------------------------
# C-15: Provider errors are normalized
# ----------------------------------------------------------------------------
def test_c15_provider_errors_normalized():
    err_timeout = normalize_http_error(408, "timed out", "p1", "m1")
    assert err_timeout.category == ProviderErrorCategory.TIMEOUT
    assert err_timeout.retryable is True

    err_rate = normalize_http_error(429, "too many requests", "p1", "m1")
    assert err_rate.category == ProviderErrorCategory.RATE_LIMIT
    assert err_rate.retryable is True


# ----------------------------------------------------------------------------
# C-16: Human fallback differs from global AI failure
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_c16_human_fallback_differs_from_technical_failure():
    storage = SQLitePromptEngineStorage(":memory:")
    session_engine = SessionEngine(storage, storage)
    provider = MockProvider()
    engine = ExecutionEngine(
        session_engine=session_engine,
        prompt_case_repo=storage,
        endpoint_catalog_repo=storage,
        variable_catalog_repo=storage,
        audit_repo=storage,
        dispatcher=OperationDispatcher(storage),
        provider=provider,
    )

    # Semantic human fallback
    provider.set_custom_handler(
        lambda req: ProviderExecutionResponse(execution_id=req.execution_id, status="SUCCESS", content="Переключаю на оператора")
    )
    outcome_hf = await engine.execute_attempt("conv_c16_1", message="Help")
    assert outcome_hf.status == ExecutionOutcomeType.HUMAN_FALLBACK

    # Technical failure
    provider.set_custom_handler(
        lambda req: ProviderExecutionResponse(execution_id=req.execution_id, status="FAILURE", error_message="Crash")
    )
    outcome_tf = await engine.execute_attempt("conv_c16_2", message="Help")
    assert outcome_tf.status == ExecutionOutcomeType.TECHNICAL_ERROR


# ----------------------------------------------------------------------------
# C-17: Concurrent session updates cannot silently overwrite state
# ----------------------------------------------------------------------------
def test_c17_concurrent_session_updates_cannot_overwrite_state():
    storage = SQLitePromptEngineStorage(":memory:")
    sess = AISession(session_id="sess_c17", conversation_id="conv_c17")
    storage.save_session(sess)

    # Exec 1 succeeds with observed version 1
    s1 = storage.get_by_id("sess_c17")
    s1.session_state = {"key": "val1"}
    storage.update_with_optimistic_lock(s1, expected_version=1)

    # Exec 2 fails with stale version 1
    s2 = storage.get_by_id("sess_c17")
    s2.session_state = {"key": "val2_stale"}
    with pytest.raises(SessionConcurrencyError):
        storage.update_with_optimistic_lock(s2, expected_version=1)


# ----------------------------------------------------------------------------
# C-18: Cache is non-authoritative
# ----------------------------------------------------------------------------
def test_c18_cache_is_non_authoritative():
    storage = SQLitePromptEngineStorage(":memory:")
    storage.save_endpoint_entry(EndpointCatalogEntry(operation_id="op1", description="desc"))
    cache = CompiledCatalogCache()

    xml1 = cache.get_compiled_endpoint_catalog(storage.list_endpoint_entries(), "rev_1")
    # Evict / invalidate cache
    cache.invalidate()
    # Re-fetches faithfully from persistent store
    xml2 = cache.get_compiled_endpoint_catalog(storage.list_endpoint_entries(), "rev_1")
    assert xml1 == xml2


# ----------------------------------------------------------------------------
# C-19: Audit identifies effective configuration
# ----------------------------------------------------------------------------
def test_c19_audit_identifies_effective_configuration():
    storage = SQLitePromptEngineStorage(":memory:")
    audit = AuditRecord(
        execution_id="exec_c19",
        case_key="resolve_user_request",
        prompt_version_number=1,
        prompt_version_hash="00" * 32,
        effective_configuration={"temperature": 0.5, "timeout": 5000},
        endpoint_catalog_revision="rev_10",
        variable_catalog_revision="rev_20",
    )
    storage.record_audit(audit)

    retrieved = storage.get_by_execution_id("exec_c19")
    assert retrieved.effective_configuration == {"temperature": 0.5, "timeout": 5000}
    assert retrieved.endpoint_catalog_revision == "rev_10"
    assert retrieved.variable_catalog_revision == "rev_20"


# ----------------------------------------------------------------------------
# C-20: Historical execution can be reconstructed
# ----------------------------------------------------------------------------
def test_c20_historical_execution_reconstruction():
    storage = SQLitePromptEngineStorage(":memory:")
    audit = AuditRecord(
        execution_id="exec_c20",
        case_key="test_case",
        prompt_version_number=2,
        prompt_version_hash="aa" * 32,
        inputs={"msg": "hello"},
        outputs={"text": "world"},
        telemetry={"latency_ms": 50.0},
    )
    storage.record_audit(audit)

    rec = storage.get_by_execution_id("exec_c20")
    assert rec.inputs == {"msg": "hello"}
    assert rec.outputs == {"text": "world"}
    assert rec.prompt_version_hash == "aa" * 32


# ----------------------------------------------------------------------------
# C-21: Shared transport is inherited from Host Project
# ----------------------------------------------------------------------------
def test_c21_shared_transport_inherited_from_host_project():
    from srtrsh_prompt_engine.api.app import app
    # Verifies FastAPI app inherits Host standard transport routes
    paths = app.openapi()["paths"]
    assert "/v1/runtime/execute" in paths
    assert "/v1/manager/prompt-cases" in paths


# ----------------------------------------------------------------------------
# C-22: Host implementation technology is not prescribed
# ----------------------------------------------------------------------------
def test_c22_host_technology_not_prescribed():
    # Demonstrates storage layer runs on InMemory as well as SQLite without external driver dependency
    in_mem = InMemoryPromptCaseRepository()
    sqlite = SQLitePromptEngineStorage(":memory:")
    case = PromptCase(case_key="free_tech")
    in_mem.save(case)
    sqlite.save_case(case)
    assert in_mem.get_by_key("free_tech") is not None
    assert sqlite.get_by_key("free_tech") is not None


# ----------------------------------------------------------------------------
# C-23: Runtime outcomes have explicit semantics
# ----------------------------------------------------------------------------
def test_c23_runtime_outcomes_explicit_semantics():
    valid_outcomes = {o.value for o in ExecutionOutcomeType}
    assert valid_outcomes == {
        "COMPLETED",
        "NEEDS_USER_INPUT",
        "INTERACTIVE_RESPONSE",
        "HUMAN_FALLBACK",
        "LIMIT_EXHAUSTED",
        "TECHNICAL_ERROR",
    }


# ----------------------------------------------------------------------------
# C-24: Manager operations enforce lifecycle/concurrency semantics
# ----------------------------------------------------------------------------
def test_c24_manager_operations_enforce_lifecycle_semantics():
    case = PromptCase(case_key="mgr_case")
    v1 = case.create_version(_sample_defn("v1"))
    case.publish_version(1)
    # Re-publishing draft creates atomic transition of v1 to DEPRECATED
    v2 = case.create_version(_sample_defn("v2"))
    case.publish_version(2)
    assert case.get_version(1).state == VersionState.DEPRECATED
    assert case.get_version(2).state == VersionState.PUBLISHED


# ----------------------------------------------------------------------------
# C-25: Continuation window does not bound historical context
# ----------------------------------------------------------------------------
def test_c25_continuation_window_does_not_bound_history():
    storage = SQLitePromptEngineStorage(":memory:")
    session_engine = SessionEngine(storage, storage, default_continuation_window_seconds=100)

    # Session 1
    s1 = session_engine.get_or_create_session("conv_c25")
    session_engine.record_message(s1.session_id, "conv_c25", "customer", "Message in session 1")

    # Time passes beyond 100s window
    s1.updated_at = datetime.now(timezone.utc) - timedelta(seconds=200)
    storage.save_session(s1)

    # Session 2 created
    s2 = session_engine.get_or_create_session("conv_c25")
    assert s2.session_id != s1.session_id

    # History from session 1 is still selected and available!
    hist = session_engine.select_history_context("conv_c25", s2.session_id, recent_messages_limit=5, previous_sessions_limit=1)
    assert any("Message in session 1" in m.content for m in hist)


# ----------------------------------------------------------------------------
# C-26: Persistent history is retained independently from context pruning
# ----------------------------------------------------------------------------
def test_c26_persistent_history_retained_independent_of_context_pruning():
    storage = SQLitePromptEngineStorage(":memory:")
    session_engine = SessionEngine(storage, storage)
    s = session_engine.get_or_create_session("conv_c26")

    # Add 20 messages
    for i in range(20):
        session_engine.record_message(s.session_id, "conv_c26", "customer", f"Msg {i}")

    # Rolling window selects only 3
    selected = session_engine.select_history_context("conv_c26", s.session_id, recent_messages_limit=3, previous_sessions_limit=0)
    assert len(selected) == 3

    # Durable persistence still retains all 20!
    durable = storage.get_messages_for_session(s.session_id)
    assert len(durable) == 20


# ----------------------------------------------------------------------------
# C-27: Offline localization generation is stateless with Host persistence
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_c27_offline_localization_stateless_with_host_persistence(client):
    payload = {
        "resourceKey": "ui.save_button",
        "targetLocale": "uz",
        "sourceText": "Сохранить",
    }
    res = await client.post("/v1/localization/offline/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "generatedText" in data

    # Prompt Engine does NOT persist this in its online template store
    app_storage = client._transport.app.state.storage
    assert app_storage.get_template("ui.save_button", "uz") is None


# ----------------------------------------------------------------------------
# C-28: Provider operation calls normalized; native formats do not leak
# ----------------------------------------------------------------------------
def test_c28_provider_operation_calls_normalized_native_formats_do_not_leak():
    call = NormalizedOperationCall(
        call_id="call_99",
        operation_id="findAvailableSlots",
        arguments={"master_id": "M1"},
    )
    assert call.call_id == "call_99"
    assert call.operation_id == "findAvailableSlots"
    assert call.arguments == {"master_id": "M1"}
    # No OpenAI or Gemini native wrapper types
    assert not hasattr(call, "function_call")
    assert not hasattr(call, "tool_call_id")


# ----------------------------------------------------------------------------
# C-29: Full catalogs compiled to compact deterministic XML
# ----------------------------------------------------------------------------
def test_c29_full_catalogs_compiled_to_compact_deterministic_xml():
    eps = [EndpointCatalogEntry(operation_id="findMasters", description="Find masters", input_variables=["service_id!"])]
    vars_list = [VariableCatalogEntry(variable_id="service_id", type=VariableType.STRING, description="Service ID")]

    xml = CatalogCompiler.compile_full_catalog(eps, "rev_1", vars_list, "rev_1")
    assert '<endpoint_catalog revision="rev_1">' in xml
    assert '<operation id="findMasters">' in xml
    assert '<input>service_id!</input>' in xml
    assert '<variables_metadata revision="rev_1">' in xml
    assert '<variable id="service_id" type="string">Service ID</variable>' in xml


# ----------------------------------------------------------------------------
# C-30: Compiled catalog cache is invalidated by catalog revision change
# ----------------------------------------------------------------------------
def test_c30_catalog_cache_invalidated_by_revision_change():
    cache = CompiledCatalogCache()
    eps = [EndpointCatalogEntry(operation_id="opA", description="desc A")]
    xml_r1 = cache.get_compiled_endpoint_catalog(eps, "rev_1")
    xml_r2 = cache.get_compiled_endpoint_catalog(eps, "rev_2")
    assert 'revision="rev_1"' in xml_r1
    assert 'revision="rev_2"' in xml_r2
    assert xml_r1 != xml_r2


# ----------------------------------------------------------------------------
# C-31: Organization System Prompt is XML-authored Prompt Version content
# ----------------------------------------------------------------------------
def test_c31_organization_system_prompt_xml_authored_content():
    case = PromptCase(case_key="organization_system")
    defn = HashedVersionDefinition(
        template=Template(
            system_prompt="<organization_system><rules><rule>Politeness</rule></rules></organization_system>",
            user_prompt="",
        ),
        schema_contract=SchemaContract(),
        execution_configuration=ExecutionConfiguration(),
    )
    v1 = case.create_version(defn)
    case.publish_version(1)

    assert "<organization_system>" in v1.definition.template.system_prompt
    assert v1.version_hash is not None
    assert case.get_published_version().case_key == "organization_system"
