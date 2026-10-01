"""Phase 1: Domain & DB Baseline Tests.

Verifies:
- Prompt Case and Prompt Version lifecycle
- Invariant AMB-01: Single PUBLISHED version with atomic deprecation of prior published version
- Immutability of published version definitions
- Closed variable type enum
- Endpoint Catalog safety boundary (READ-only operations in MVP)
- AISession initialized to session_version=1
- Optimistic locking compare-and-update behavior in both InMemory and SQLite storage
- Rejection of stale session versions (Master section 66, C-17)
- OnlineTemplate deterministic validation and rendering (C-13)
- AuditRecord completeness and reproducibility foundations
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import pytest

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
from srtrsh_prompt_engine.storage.repository import (
    InMemoryAISessionRepository,
    InMemoryEndpointCatalogRepository,
    InMemoryOnlineTemplateRepository,
    InMemoryPromptCaseRepository,
    InMemorySessionHistoryRepository,
    InMemoryVariableCatalogRepository,
    SQLitePromptEngineStorage,
)


def _make_sample_definition(system_text: str = "System instructions"):
    return HashedVersionDefinition(
        template=Template(
            system_prompt=system_text,
            user_prompt="User input: {{query}}",
        ),
        variables=[
            VariableDefinition(name="query", type="string", required=True),
        ],
        schema_contract=SchemaContract(
            schema_type="JSON_SCHEMA",
            definition={"type": "object"},
        ),
        execution_configuration=ExecutionConfiguration(
            temperature=0.2,
            timeout_duration=4000,
        ),
        override_policy=RuntimeOverridePolicy(
            allowed_parameters={"temperature"},
        ),
    )


# ============================================================================
# 1. PromptCase & PromptVersion Lifecycle & Invariants
# ============================================================================


def test_prompt_case_identity_and_version_creation():
    case = PromptCase(case_key="resolve_user_request", description="Resolve booking intent")
    assert case.case_key == "resolve_user_request"
    assert case.identity == "resolve_user_request"
    assert case.published_version_number is None

    defn = _make_sample_definition()
    v1 = case.create_version(defn, author="developer1")
    assert v1.version_number == 1
    assert v1.case_key == "resolve_user_request"
    assert v1.state == VersionState.DRAFT
    assert v1.version_hash is None
    assert v1.author == "developer1"


def test_single_published_version_invariant_and_atomic_deprecation():
    """Invariant AMB-01: At most one version of a given Prompt Case may be in PUBLISHED

    status at any time. Transitioning a draft to PUBLISHED atomically transitions
    any previously PUBLISHED version of that caseKey to DEPRECATED.
    """
    case = PromptCase(case_key="organization_system")

    # Create v1 and publish
    v1 = case.create_version(_make_sample_definition("System v1"))
    v1_pub = case.publish_version(1)
    assert v1_pub.state == VersionState.PUBLISHED
    assert case.published_version_number == 1
    assert v1_pub.version_hash is not None

    # Create v2 as draft
    v2 = case.create_version(_make_sample_definition("System v2"))
    assert v2.version_number == 2
    assert v2.state == VersionState.DRAFT
    # v1 is still published
    assert case.get_version(1).state == VersionState.PUBLISHED

    # Publish v2 -> v1 must be atomically transitioned to DEPRECATED
    v2_pub = case.publish_version(2)
    assert v2_pub.state == VersionState.PUBLISHED
    assert case.published_version_number == 2
    assert case.get_version(1).state == VersionState.DEPRECATED

    # Create v3 as draft and publish -> v2 becomes DEPRECATED, v3 becomes PUBLISHED
    v3 = case.create_version(_make_sample_definition("System v3"))
    v3_pub = case.publish_version(3)
    assert v3_pub.state == VersionState.PUBLISHED
    assert case.published_version_number == 3
    assert case.get_version(2).state == VersionState.DEPRECATED
    assert case.get_version(1).state == VersionState.DEPRECATED

    # Invariant: Deprecated versions cannot be re-published
    with pytest.raises(ValueError, match="Cannot publish a deprecated version"):
        case.publish_version(1)
    with pytest.raises(ValueError, match="Cannot publish a deprecated version"):
        case.publish_version(2)


def test_published_version_immutability():
    """Published versions cannot be modified in place."""
    case = PromptCase(case_key="find_services")
    v1 = case.create_version(_make_sample_definition("Original prompt"))
    v1_pub = case.publish_version(1)
    assert v1_pub.state == VersionState.PUBLISHED

    # Attempting to reassign definition on published version must raise ValueError
    new_defn = _make_sample_definition("Modified prompt")
    with pytest.raises(ValueError, match="Published versions are immutable"):
        v1_pub.definition = new_defn


def test_distinct_version_number_and_hash():
    """C-03: Version number and Prompt Version Hash are distinct identities."""
    case = PromptCase(case_key="appointment_reminder")
    defn = _make_sample_definition()
    v1 = case.create_version(defn)
    case.publish_version(1)

    assert v1.version_number == 1
    assert isinstance(v1.version_hash, str)
    assert len(v1.version_hash) == 64
    assert v1.version_hash.isalnum()


# ============================================================================
# 2. Endpoint Catalog & Variable Catalog Invariants
# ============================================================================


def test_endpoint_catalog_safety_boundary():
    """C-11: Safety boundary - write operations unavailable in MVP."""
    # Read-only operation succeeds
    entry_read = EndpointCatalogEntry(
        operation_id="findAvailableSlots",
        description="Find available appointment slots",
        input_variables=["master_id!", "service_id!"],
        output_variables=["available_slots"],
        is_read_only=True,
    )
    assert entry_read.operation_id == "findAvailableSlots"
    assert entry_read.is_read_only is True

    # Write operation fails validation
    with pytest.raises(ValueError, match="Write operation 'createBooking' is not permitted"):
        EndpointCatalogEntry(
            operation_id="createBooking",
            description="Create appointment booking",
            input_variables=["master_id!", "customer_id!"],
            output_variables=["booking_id"],
            is_read_only=False,
        )


def test_variable_catalog_closed_type_enum():
    """AMB-02: Variable catalog type system matches the closed enum."""
    valid_types = [
        "string",
        "integer",
        "number",
        "boolean",
        "datetime",
        "array",
        "object",
        "STRING",
        "INTEGER",
    ]
    for vt in valid_types:
        entry = VariableCatalogEntry(
            variable_id=f"var_{vt.lower()}",
            type=VariableType(vt),
            description=f"Variable of type {vt}",
        )
        assert entry.type in VariableType

    # Unknown type raises ValueError
    with pytest.raises(ValueError, match="Invalid variable type 'custom_map'"):
        VariableCatalogEntry(
            variable_id="invalid_var",
            type="custom_map",  # type: ignore
            description="Invalid custom type",
        )


# ============================================================================
# 3. AI Session Initialization & Concurrency Invariants
# ============================================================================


def test_ai_session_initialization():
    """Master section 66: Persistent AI Session record MUST contain a non-null

    monotonically increasing session version, initialized to 1.
    """
    sess = AISession(conversation_id="conv_100")
    assert sess.session_version == 1
    assert sess.is_active is True
    assert sess.session_state == {}
    assert sess.business_references == {}
    assert sess.interaction_results == []

    # Invalid session version < 1 raises ValueError
    with pytest.raises(ValueError, match="session_version must be a non-null positive integer >= 1"):
        AISession(conversation_id="conv_bad", session_version=0)


def test_optimistic_locking_in_memory_repository():
    """Verify Master section 66 verification protocol on InMemoryAISessionRepository.

    1. Create Session with initial session version = 1.
    2. Start two executions observing session version 1.
    3. Allow both to attempt state update based on version 1.
    4. Verify exactly one succeeds.
    5. Verify successful update advances version to 2.
    6. Verify other execution is rejected as concurrency conflict.
    7. Verify rejected execution does not overwrite state.
    8. Verify final state matches successful update.
    9. Repeat for later version (v2 -> v3).
    """
    repo = InMemoryAISessionRepository()
    sess = AISession(session_id="sess_concurrency_test", conversation_id="conv_c17")
    repo.save(sess)

    # Both read version 1
    exec1_read = repo.get_by_id("sess_concurrency_test")
    exec2_read = repo.get_by_id("sess_concurrency_test")
    assert exec1_read.session_version == 1
    assert exec2_read.session_version == 1

    # Exec 1 modifies state and updates with expected_version = 1
    exec1_read.session_state = {"service_id": "S123"}
    updated1 = repo.update_with_optimistic_lock(exec1_read, expected_version=1)
    assert updated1.session_version == 2

    # Exec 2 tries to update with stale expected_version = 1 -> Must raise SessionConcurrencyError
    exec2_read.session_state = {"service_id": "S456_OVERWRITE_ATTEMPT"}
    with pytest.raises(SessionConcurrencyError, match="expected version 1, but found 2"):
        repo.update_with_optimistic_lock(exec2_read, expected_version=1)

    # Persisted state must match Exec 1, NOT Exec 2
    final_sess = repo.get_by_id("sess_concurrency_test")
    assert final_sess.session_version == 2
    assert final_sess.session_state == {"service_id": "S123"}

    # Repeat with later version (v2 -> v3)
    final_sess.session_state["master_id"] = "M42"
    updated2 = repo.update_with_optimistic_lock(final_sess, expected_version=2)
    assert updated2.session_version == 3

    # Attempting with stale version 2 fails
    final_sess.session_state["master_id"] = "STALE_M"
    with pytest.raises(SessionConcurrencyError, match="expected version 2, but found 3"):
        repo.update_with_optimistic_lock(final_sess, expected_version=2)


def test_optimistic_locking_sqlite_repository():
    """Verify Master section 66 verification protocol on SQLitePromptEngineStorage."""
    storage = SQLitePromptEngineStorage(":memory:")
    sess = AISession(session_id="sess_sqlite_concurrency", conversation_id="conv_sqlite")
    storage.save(sess)

    read1 = storage.get_by_id("sess_sqlite_concurrency")
    read2 = storage.get_by_id("sess_sqlite_concurrency")
    assert read1.session_version == 1
    assert read2.session_version == 1

    # Exec 1 succeeds
    read1.session_state = {"customer_id": "C777"}
    updated1 = storage.update_with_optimistic_lock(read1, expected_version=1)
    assert updated1.session_version == 2

    # Exec 2 fails
    read2.session_state = {"customer_id": "STALE_C"}
    with pytest.raises(SessionConcurrencyError, match="expected version 1, but found 2"):
        storage.update_with_optimistic_lock(read2, expected_version=1)

    # Final persisted state check
    persisted = storage.get_by_id("sess_sqlite_concurrency")
    assert persisted.session_version == 2
    assert persisted.session_state == {"customer_id": "C777"}

    # Repeat for v2 -> v3
    persisted.session_state["confirmed"] = True
    updated2 = storage.update_with_optimistic_lock(persisted, expected_version=2)
    assert updated2.session_version == 3
    assert storage.get_by_id("sess_sqlite_concurrency").session_version == 3


def test_multi_threaded_concurrency_race_condition():
    """Race condition test: Multiple threads race to increment session version.

    Only one thread must succeed; all others must receive SessionConcurrencyError.
    """
    storage = SQLitePromptEngineStorage(":memory:")
    sess = AISession(session_id="race_sess", conversation_id="conv_race")
    storage.save(sess)

    successes = []
    conflicts = []

    def attempt_update(worker_id: int):
        sess_copy = storage.get_by_id("race_sess")
        # All workers observed version 1
        sess_copy.session_state = {f"worker_{worker_id}": True}
        try:
            storage.update_with_optimistic_lock(sess_copy, expected_version=1)
            successes.append(worker_id)
        except SessionConcurrencyError:
            conflicts.append(worker_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(attempt_update, i) for i in range(8)]
        for f in futures:
            f.result()

    # Exactly one thread succeeds
    assert len(successes) == 1
    assert len(conflicts) == 7
    # Version advanced to exactly 2
    final = storage.get_by_id("race_sess")
    assert final.session_version == 2


# ============================================================================
# 4. OnlineTemplate Deterministic Validation & Rendering (C-13)
# ============================================================================


def test_online_template_validation_and_rendering():
    tmpl = OnlineTemplate(
        case_key="appointment_reminder",
        locale="ru",
        template_text="$client_identity$, у вас $date_human_readable$ запись к мастеру $master_name$ в $time$.",
        required_variables=["client_identity", "date_human_readable", "master_name", "time"],
    )

    vars_extracted = tmpl.extract_variables()
    assert vars_extracted == {"client_identity", "date_human_readable", "master_name", "time"}

    # Complete valid variables
    valid, missing = tmpl.validate_variables(
        {
            "client_identity": "Иван",
            "date_human_readable": "завтра",
            "master_name": "Алексей",
            "time": "14:00",
        }
    )
    assert valid is True
    assert missing == []

    rendered = tmpl.render(
        {
            "client_identity": "Иван",
            "date_human_readable": "завтра",
            "master_name": "Алексей",
            "time": "14:00",
        }
    )
    assert rendered == "Иван, у вас завтра запись к мастеру Алексей в 14:00."

    # Missing variables fails validation and render
    valid_missing, missing_vars = tmpl.validate_variables(
        {"client_identity": "Иван", "date_human_readable": "завтра"}
    )
    assert valid_missing is False
    assert "master_name" in missing_vars
    assert "time" in missing_vars

    with pytest.raises(ValueError, match="Cannot render template: missing variables"):
        tmpl.render({"client_identity": "Иван"})


# ============================================================================
# 5. AuditRecord Completeness & Persistence
# ============================================================================


def test_audit_record_round_trip():
    storage = SQLitePromptEngineStorage(":memory:")
    audit = AuditRecord(
        execution_id="exec_999",
        session_id="sess_123",
        conversation_id="conv_456",
        case_key="organization_system",
        prompt_version_number=1,
        prompt_version_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        endpoint_catalog_revision="rev_101",
        variable_catalog_revision="rev_202",
        effective_configuration={"temperature": 0.0},
        inputs={"query": "test"},
        outputs={"result": "ok"},
        telemetry={"latency_ms": 120.5},
    )
    storage.record_audit(audit)

    retrieved = storage.get_by_execution_id("exec_999")
    assert retrieved is not None
    assert retrieved.execution_id == "exec_999"
    assert retrieved.session_id == "sess_123"
    assert retrieved.prompt_version_number == 1
    assert retrieved.prompt_version_hash == audit.prompt_version_hash
    assert retrieved.endpoint_catalog_revision == "rev_101"
    assert retrieved.variable_catalog_revision == "rev_202"
    assert retrieved.outputs == {"result": "ok"}
    assert retrieved.telemetry["latency_ms"] == 120.5
