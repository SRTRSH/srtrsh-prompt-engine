"""Persistence and repository layer for Prompt Engine.

Normative references:
WikiDocs/14-prompt-engine/03-domain-model/
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/06-session-concurrency.md
WikiDocs/14-prompt-engine/15-persistence/
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from srtrsh_prompt_engine.domain.models import (
    AISession,
    AuditRecord,
    EndpointCatalogEntry,
    OnlineTemplate,
    PromptCase,
    SessionConcurrencyError,
    SessionMessage,
    VariableCatalogEntry,
    VariableType,
)


# ============================================================================
# Prompt Case Repositories
# ============================================================================


class PromptCaseRepository(ABC):
    """Abstract interface for PromptCase aggregate persistence."""

    @abstractmethod
    def get_by_key(self, case_key: str) -> Optional[PromptCase]:
        """Retrieve PromptCase aggregate by its caseKey."""
        ...

    def get_by_identity(self, identity: str) -> Optional[PromptCase]:
        """Backward compatibility alias for get_by_key."""
        return self.get_by_key(identity)

    @abstractmethod
    def save(self, case: PromptCase) -> None:
        """Persist or update PromptCase aggregate."""
        ...

    @abstractmethod
    def list_all(self) -> List[PromptCase]:
        """List all registered prompt cases."""
        ...


# Backward compatibility alias
PromptRepository = PromptCaseRepository


class InMemoryPromptCaseRepository(PromptCaseRepository):
    """Thread-safe in-memory storage for prompt cases."""

    def __init__(self) -> None:
        self._cases: Dict[str, PromptCase] = {}
        self._lock = threading.Lock()

    def get_by_key(self, case_key: str) -> Optional[PromptCase]:
        with self._lock:
            case = self._cases.get(case_key)
            return case.model_copy(deep=True) if case else None

    def save(self, case: PromptCase) -> None:
        with self._lock:
            self._cases[case.case_key] = case.model_copy(deep=True)

    def list_all(self) -> List[PromptCase]:
        with self._lock:
            return [c.model_copy(deep=True) for c in self._cases.values()]


# Backward compatibility alias
InMemoryPromptRepository = InMemoryPromptCaseRepository


# ============================================================================
# Endpoint Catalog Repositories
# ============================================================================


class EndpointCatalogRepository(ABC):
    """Abstract interface for Endpoint Catalog entries."""

    @abstractmethod
    def get_entry(self, operation_id: str) -> Optional[EndpointCatalogEntry]:
        ...

    @abstractmethod
    def list_entries(self) -> List[EndpointCatalogEntry]:
        ...

    @abstractmethod
    def save_entry(self, entry: EndpointCatalogEntry) -> None:
        ...

    @abstractmethod
    def delete_entry(self, operation_id: str) -> None:
        ...

    @abstractmethod
    def get_current_revision(self) -> str:
        ...

    @abstractmethod
    def set_current_revision(self, revision: str) -> None:
        ...


class InMemoryEndpointCatalogRepository(EndpointCatalogRepository):
    """Thread-safe in-memory storage for endpoint catalog entries."""

    def __init__(self, revision: str = "rev_1") -> None:
        self._entries: Dict[str, EndpointCatalogEntry] = {}
        self._revision: str = revision
        self._lock = threading.Lock()

    def get_entry(self, operation_id: str) -> Optional[EndpointCatalogEntry]:
        with self._lock:
            entry = self._entries.get(operation_id)
            return entry.model_copy(deep=True) if entry else None

    def list_entries(self) -> List[EndpointCatalogEntry]:
        with self._lock:
            # Deterministically sorted by operationId ascending
            sorted_entries = sorted(self._entries.values(), key=lambda e: e.operation_id)
            return [e.model_copy(deep=True) for e in sorted_entries]

    def save_entry(self, entry: EndpointCatalogEntry) -> None:
        with self._lock:
            self._entries[entry.operation_id] = entry.model_copy(deep=True)

    def delete_entry(self, operation_id: str) -> None:
        with self._lock:
            self._entries.pop(operation_id, None)

    def get_current_revision(self) -> str:
        with self._lock:
            return self._revision

    def set_current_revision(self, revision: str) -> None:
        with self._lock:
            self._revision = revision


# ============================================================================
# Variable Catalog Repositories
# ============================================================================


class VariableCatalogRepository(ABC):
    """Abstract interface for Variable Catalog entries."""

    @abstractmethod
    def get_entry(self, variable_id: str) -> Optional[VariableCatalogEntry]:
        ...

    @abstractmethod
    def list_entries(self) -> List[VariableCatalogEntry]:
        ...

    @abstractmethod
    def save_entry(self, entry: VariableCatalogEntry) -> None:
        ...

    @abstractmethod
    def delete_entry(self, variable_id: str) -> None:
        ...

    @abstractmethod
    def get_current_revision(self) -> str:
        ...

    @abstractmethod
    def set_current_revision(self, revision: str) -> None:
        ...


class InMemoryVariableCatalogRepository(VariableCatalogRepository):
    """Thread-safe in-memory storage for variable catalog entries."""

    def __init__(self, revision: str = "rev_1") -> None:
        self._entries: Dict[str, VariableCatalogEntry] = {}
        self._revision: str = revision
        self._lock = threading.Lock()

    def get_entry(self, variable_id: str) -> Optional[VariableCatalogEntry]:
        with self._lock:
            entry = self._entries.get(variable_id)
            return entry.model_copy(deep=True) if entry else None

    def list_entries(self) -> List[VariableCatalogEntry]:
        with self._lock:
            # Deterministically sorted by variableId ascending
            sorted_entries = sorted(self._entries.values(), key=lambda e: e.variable_id)
            return [e.model_copy(deep=True) for e in sorted_entries]

    def save_entry(self, entry: VariableCatalogEntry) -> None:
        with self._lock:
            self._entries[entry.variable_id] = entry.model_copy(deep=True)

    def delete_entry(self, variable_id: str) -> None:
        with self._lock:
            self._entries.pop(variable_id, None)

    def get_current_revision(self) -> str:
        with self._lock:
            return self._revision

    def set_current_revision(self, revision: str) -> None:
        with self._lock:
            self._revision = revision


# ============================================================================
# AI Session Repositories & Concurrency Control
# ============================================================================


class AISessionRepository(ABC):
    """Abstract interface for AI Session persistence with optimistic concurrency."""

    @abstractmethod
    def get_by_id(self, session_id: str) -> Optional[AISession]:
        ...

    @abstractmethod
    def get_active_by_conversation_id(self, conversation_id: str) -> Optional[AISession]:
        ...

    @abstractmethod
    def save(self, session: AISession) -> None:
        """Create a new session (initialized to session_version=1)."""
        ...

    @abstractmethod
    def update_with_optimistic_lock(
        self, session: AISession, expected_version: int
    ) -> AISession:
        """Atomically update session if stored session_version matches expected_version.

        Advances session_version by exactly 1 on success.
        Raises SessionConcurrencyError if stored session_version != expected_version.
        """
        ...


class InMemoryAISessionRepository(AISessionRepository):
    """Thread-safe in-memory AI Session store with strict optimistic locking."""

    def __init__(self) -> None:
        self._sessions: Dict[str, AISession] = {}
        self._lock = threading.Lock()

    def get_by_id(self, session_id: str) -> Optional[AISession]:
        with self._lock:
            sess = self._sessions.get(session_id)
            return sess.model_copy(deep=True) if sess else None

    def get_active_by_conversation_id(self, conversation_id: str) -> Optional[AISession]:
        with self._lock:
            for sess in self._sessions.values():
                if sess.conversation_id == conversation_id and sess.is_active:
                    return sess.model_copy(deep=True)
            return None

    def save(self, session: AISession) -> None:
        with self._lock:
            # Enforce initial session_version = 1
            if session.session_version is None or session.session_version < 1:
                session.session_version = 1
            self._sessions[session.session_id] = session.model_copy(deep=True)

    def update_with_optimistic_lock(
        self, session: AISession, expected_version: int
    ) -> AISession:
        with self._lock:
            stored = self._sessions.get(session.session_id)
            if not stored:
                raise ValueError(f"Session {session.session_id} does not exist.")
            if stored.session_version != expected_version:
                raise SessionConcurrencyError(
                    f"Session concurrency conflict on {session.session_id}: "
                    f"expected version {expected_version}, but found {stored.session_version}"
                )
            # Advance version by exactly one
            session.session_version = expected_version + 1
            session.updated_at = datetime.now(timezone.utc)
            self._sessions[session.session_id] = session.model_copy(deep=True)
            return session.model_copy(deep=True)


# ============================================================================
# Conversation History Repositories
# ============================================================================


class SessionHistoryRepository(ABC):
    """Durable conversation history repository."""

    @abstractmethod
    def append_message(self, message: SessionMessage) -> None:
        ...

    @abstractmethod
    def get_messages_for_session(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[SessionMessage]:
        ...

    @abstractmethod
    def get_messages_for_conversation(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[SessionMessage]:
        ...

    @abstractmethod
    def get_previous_sessions(
        self, conversation_id: str, current_session_id: str, limit: int = 10
    ) -> List[str]:
        ...


class InMemorySessionHistoryRepository(SessionHistoryRepository):
    """Thread-safe in-memory session history store."""

    def __init__(self) -> None:
        self._messages: List[SessionMessage] = []
        self._lock = threading.Lock()

    def append_message(self, message: SessionMessage) -> None:
        with self._lock:
            self._messages.append(message.model_copy(deep=True))

    def get_messages_for_session(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[SessionMessage]:
        with self._lock:
            msgs = [m.model_copy(deep=True) for m in self._messages if m.session_id == session_id]
            msgs.sort(key=lambda m: m.created_at)
            if limit is not None and limit > 0:
                msgs = msgs[-limit:]
            return msgs

    def get_messages_for_conversation(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[SessionMessage]:
        with self._lock:
            msgs = [
                m.model_copy(deep=True)
                for m in self._messages
                if m.conversation_id == conversation_id
            ]
            msgs.sort(key=lambda m: m.created_at)
            if limit is not None and limit > 0:
                msgs = msgs[-limit:]
            return msgs

    def get_previous_sessions(
        self, conversation_id: str, current_session_id: str, limit: int = 10
    ) -> List[str]:
        with self._lock:
            seen = []
            for m in self._messages:
                if m.conversation_id == conversation_id and m.session_id != current_session_id:
                    if m.session_id not in seen:
                        seen.append(m.session_id)
            return seen[-limit:]


# ============================================================================
# Online Localization Template Repositories
# ============================================================================


class OnlineTemplateRepository(ABC):
    """Abstract interface for Online Localization Templates."""

    @abstractmethod
    def get_template(self, case_key: str, locale: str) -> Optional[OnlineTemplate]:
        ...

    @abstractmethod
    def save_template(self, template: OnlineTemplate) -> None:
        ...

    @abstractmethod
    def list_templates(self, case_key: Optional[str] = None) -> List[OnlineTemplate]:
        ...


class InMemoryOnlineTemplateRepository(OnlineTemplateRepository):
    """Thread-safe in-memory store for online templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, OnlineTemplate] = {}
        self._lock = threading.Lock()

    def _key(self, case_key: str, locale: str) -> str:
        return f"{case_key}::{locale.lower()}"

    def get_template(self, case_key: str, locale: str) -> Optional[OnlineTemplate]:
        with self._lock:
            t = self._templates.get(self._key(case_key, locale))
            return t.model_copy(deep=True) if t else None

    def save_template(self, template: OnlineTemplate) -> None:
        with self._lock:
            k = self._key(template.case_key, template.locale)
            self._templates[k] = template.model_copy(deep=True)

    def list_templates(self, case_key: Optional[str] = None) -> List[OnlineTemplate]:
        with self._lock:
            res = [
                t.model_copy(deep=True)
                for t in self._templates.values()
                if case_key is None or t.case_key == case_key
            ]
            return res


# ============================================================================
# Audit Record Repositories
# ============================================================================


class AuditRepository(ABC):
    """Abstract interface for execution audit persistence."""

    @abstractmethod
    def record_audit(self, record: AuditRecord) -> None:
        ...

    @abstractmethod
    def get_by_execution_id(self, execution_id: str) -> Optional[AuditRecord]:
        ...

    @abstractmethod
    def list_by_session_id(self, session_id: str) -> List[AuditRecord]:
        ...


class InMemoryAuditRepository(AuditRepository):
    """Thread-safe in-memory audit store."""

    def __init__(self) -> None:
        self._records: Dict[str, AuditRecord] = {}
        self._lock = threading.Lock()

    def record_audit(self, record: AuditRecord) -> None:
        with self._lock:
            self._records[record.execution_id] = record.model_copy(deep=True)

    def get_by_execution_id(self, execution_id: str) -> Optional[AuditRecord]:
        with self._lock:
            r = self._records.get(execution_id)
            return r.model_copy(deep=True) if r else None

    def list_by_session_id(self, session_id: str) -> List[AuditRecord]:
        with self._lock:
            return [
                r.model_copy(deep=True)
                for r in self._records.values()
                if r.session_id == session_id
            ]


# ============================================================================
# SQLite Durable Storage Implementation
# ============================================================================


class SQLitePromptEngineStorage(
    PromptCaseRepository,
    EndpointCatalogRepository,
    VariableCatalogRepository,
    AISessionRepository,
    SessionHistoryRepository,
    OnlineTemplateRepository,
    AuditRepository,
):
    """ACID-compliant SQLite storage implementing all Prompt Engine repositories.

    Supports thread-safe execution, transactions, and optimistic concurrency.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        # For :memory:, check_same_thread=False allows multi-thread testing if protected by mutex
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS prompt_cases (
                    case_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS endpoint_catalog (
                    operation_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    human_description TEXT,
                    input_variables TEXT NOT NULL,
                    output_variables TEXT NOT NULL,
                    is_read_only INTEGER NOT NULL DEFAULT 1,
                    execution_constraints TEXT NOT NULL,
                    revision TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS variable_catalog (
                    variable_id TEXT PRIMARY KEY,
                    var_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    revision TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS catalog_revisions (
                    catalog_name TEXT PRIMARY KEY,
                    revision TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_sessions (
                    session_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    session_version INTEGER NOT NULL DEFAULT 1,
                    session_state TEXT NOT NULL,
                    business_references TEXT NOT NULL,
                    interaction_results TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    closed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_conv ON ai_sessions(conversation_id, is_active);

                CREATE TABLE IF NOT EXISTS session_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_msgs_session ON session_messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_msgs_conv ON session_messages(conversation_id, created_at);

                CREATE TABLE IF NOT EXISTS online_templates (
                    template_id TEXT PRIMARY KEY,
                    case_key TEXT NOT NULL,
                    locale TEXT NOT NULL,
                    template_text TEXT NOT NULL,
                    required_variables TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(case_key, locale)
                );

                CREATE TABLE IF NOT EXISTS audit_records (
                    audit_id TEXT PRIMARY KEY,
                    execution_id TEXT UNIQUE NOT NULL,
                    session_id TEXT,
                    conversation_id TEXT,
                    case_key TEXT NOT NULL,
                    prompt_version_number INTEGER NOT NULL,
                    prompt_version_hash TEXT NOT NULL,
                    endpoint_catalog_revision TEXT,
                    variable_catalog_revision TEXT,
                    effective_config TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    outputs TEXT,
                    telemetry TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # --- PromptCaseRepository ---
    def get_by_key(self, case_key: str) -> Optional[PromptCase]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT payload FROM prompt_cases WHERE case_key = ?", (case_key,))
            row = cur.fetchone()
            if row:
                data = json.loads(row["payload"])
                return PromptCase.model_validate(data)
            return None

    def save_case(self, case: PromptCase) -> None:
        with self._lock:
            cur = self._conn.cursor()
            payload = json.dumps(case.model_dump(mode="json"))
            cur.execute(
                """
                INSERT INTO prompt_cases (case_key, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(case_key) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (case.case_key, payload, case.updated_at.isoformat()),
            )
            self._conn.commit()

    def save(self, entity: Any) -> None:
        if isinstance(entity, PromptCase):
            self.save_case(entity)
        elif isinstance(entity, AISession):
            self.save_session(entity)
        else:
            raise TypeError(f"Unsupported entity type for save: {type(entity)}")

    def list_all(self) -> List[PromptCase]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT payload FROM prompt_cases ORDER BY case_key ASC")
            rows = cur.fetchall()
            return [PromptCase.model_validate(json.loads(r["payload"])) for r in rows]

    # --- EndpointCatalogRepository ---
    def get_endpoint_entry(self, operation_id: str) -> Optional[EndpointCatalogEntry]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM endpoint_catalog WHERE operation_id = ?", (operation_id,))
            row = cur.fetchone()
            if not row:
                return None
            return EndpointCatalogEntry(
                operation_id=row["operation_id"],
                description=row["description"],
                human_description=row["human_description"],
                input_variables=json.loads(row["input_variables"]),
                output_variables=json.loads(row["output_variables"]),
                is_read_only=bool(row["is_read_only"]),
                execution_constraints=json.loads(row["execution_constraints"]),
                revision=row["revision"],
            )

    def list_endpoint_entries(self) -> List[EndpointCatalogEntry]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM endpoint_catalog ORDER BY operation_id ASC")
            rows = cur.fetchall()
            return [
                EndpointCatalogEntry(
                    operation_id=r["operation_id"],
                    description=r["description"],
                    human_description=r["human_description"],
                    input_variables=json.loads(r["input_variables"]),
                    output_variables=json.loads(r["output_variables"]),
                    is_read_only=bool(r["is_read_only"]),
                    execution_constraints=json.loads(r["execution_constraints"]),
                    revision=r["revision"],
                )
                for r in rows
            ]

    def save_endpoint_entry(self, entry: EndpointCatalogEntry) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO endpoint_catalog (
                    operation_id, description, human_description, input_variables,
                    output_variables, is_read_only, execution_constraints, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    description = excluded.description,
                    human_description = excluded.human_description,
                    input_variables = excluded.input_variables,
                    output_variables = excluded.output_variables,
                    is_read_only = excluded.is_read_only,
                    execution_constraints = excluded.execution_constraints,
                    revision = excluded.revision
                """,
                (
                    entry.operation_id,
                    entry.description,
                    entry.human_description,
                    json.dumps(entry.input_variables),
                    json.dumps(entry.output_variables),
                    1 if entry.is_read_only else 0,
                    json.dumps(entry.execution_constraints),
                    entry.revision,
                ),
            )
            self._conn.commit()

    def delete_endpoint_entry(self, operation_id: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM endpoint_catalog WHERE operation_id = ?", (operation_id,))
            self._conn.commit()

    def get_endpoint_catalog_revision(self) -> str:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT revision FROM catalog_revisions WHERE catalog_name = 'endpoint_catalog'"
            )
            row = cur.fetchone()
            return row["revision"] if row else "rev_1"

    def set_endpoint_catalog_revision(self, revision: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO catalog_revisions (catalog_name, revision)
                VALUES ('endpoint_catalog', ?)
                ON CONFLICT(catalog_name) DO UPDATE SET revision = excluded.revision
                """,
                (revision,),
            )
            self._conn.commit()

    # --- VariableCatalogRepository ---
    def get_variable_entry(self, variable_id: str) -> Optional[VariableCatalogEntry]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM variable_catalog WHERE variable_id = ?", (variable_id,))
            row = cur.fetchone()
            if not row:
                return None
            return VariableCatalogEntry(
                variable_id=row["variable_id"],
                type=VariableType(row["var_type"]),
                description=row["description"],
                revision=row["revision"],
            )

    def list_variable_entries(self) -> List[VariableCatalogEntry]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM variable_catalog ORDER BY variable_id ASC")
            rows = cur.fetchall()
            return [
                VariableCatalogEntry(
                    variable_id=r["variable_id"],
                    type=VariableType(r["var_type"]),
                    description=r["description"],
                    revision=r["revision"],
                )
                for r in rows
            ]

    def save_variable_entry(self, entry: VariableCatalogEntry) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO variable_catalog (variable_id, var_type, description, revision)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(variable_id) DO UPDATE SET
                    var_type = excluded.var_type,
                    description = excluded.description,
                    revision = excluded.revision
                """,
                (entry.variable_id, entry.type.value, entry.description, entry.revision),
            )
            self._conn.commit()

    def delete_variable_entry(self, variable_id: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM variable_catalog WHERE variable_id = ?", (variable_id,))
            self._conn.commit()

    def get_variable_catalog_revision(self) -> str:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT revision FROM catalog_revisions WHERE catalog_name = 'variable_catalog'"
            )
            row = cur.fetchone()
            return row["revision"] if row else "rev_1"

    def set_variable_catalog_revision(self, revision: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO catalog_revisions (catalog_name, revision)
                VALUES ('variable_catalog', ?)
                ON CONFLICT(catalog_name) DO UPDATE SET revision = excluded.revision
                """,
                (revision,),
            )
            self._conn.commit()

    # Polymorphic catalog methods
    def save_entry(self, entry: Any) -> None:
        if isinstance(entry, EndpointCatalogEntry):
            self.save_endpoint_entry(entry)
        elif isinstance(entry, VariableCatalogEntry):
            self.save_variable_entry(entry)
        else:
            raise TypeError(f"Unknown catalog entry type: {type(entry)}")

    def get_entry(self, identifier: str) -> Any:
        res = self.get_endpoint_entry(identifier)
        if res is not None:
            return res
        return self.get_variable_entry(identifier)

    def delete_entry(self, identifier: str) -> None:
        self.delete_endpoint_entry(identifier)
        self.delete_variable_entry(identifier)

    def list_entries(self) -> List[Any]:
        return self.list_endpoint_entries()

    def get_current_revision(self) -> str:
        return self.get_endpoint_catalog_revision()

    def set_current_revision(self, revision: str) -> None:
        self.set_endpoint_catalog_revision(revision)
        self.set_variable_catalog_revision(revision)

    # --- AISessionRepository & Optimistic Concurrency ---
    def get_by_id(self, session_id: str) -> Optional[AISession]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM ai_sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_session(row)

    def get_active_by_conversation_id(self, conversation_id: str) -> Optional[AISession]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM ai_sessions WHERE conversation_id = ? AND is_active = 1",
                (conversation_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_session(row)

    def save_session(self, session: AISession) -> None:
        with self._lock:
            cur = self._conn.cursor()
            if session.session_version is None or session.session_version < 1:
                session.session_version = 1
            cur.execute(
                """
                INSERT INTO ai_sessions (
                    session_id, conversation_id, session_version, session_state,
                    business_references, interaction_results, is_active,
                    created_at, updated_at, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    session_version = excluded.session_version,
                    session_state = excluded.session_state,
                    business_references = excluded.business_references,
                    interaction_results = excluded.interaction_results,
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at,
                    closed_at = excluded.closed_at
                """,
                (
                    session.session_id,
                    session.conversation_id,
                    session.session_version,
                    json.dumps(session.session_state),
                    json.dumps(session.business_references),
                    json.dumps(session.interaction_results),
                    1 if session.is_active else 0,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.closed_at.isoformat() if session.closed_at else None,
                ),
            )
            self._conn.commit()

    def update_with_optimistic_lock(
        self, session: AISession, expected_version: int
    ) -> AISession:
        with self._lock:
            cur = self._conn.cursor()
            new_version = expected_version + 1
            now_iso = datetime.now(timezone.utc).isoformat()
            cur.execute(
                """
                UPDATE ai_sessions
                SET session_version = ?,
                    session_state = ?,
                    business_references = ?,
                    interaction_results = ?,
                    is_active = ?,
                    updated_at = ?,
                    closed_at = ?
                WHERE session_id = ? AND session_version = ?
                """,
                (
                    new_version,
                    json.dumps(session.session_state),
                    json.dumps(session.business_references),
                    json.dumps(session.interaction_results),
                    1 if session.is_active else 0,
                    now_iso,
                    session.closed_at.isoformat() if session.closed_at else None,
                    session.session_id,
                    expected_version,
                ),
            )
            if cur.rowcount == 0:
                cur.execute(
                    "SELECT session_version FROM ai_sessions WHERE session_id = ?",
                    (session.session_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"Session {session.session_id} does not exist.")
                actual_version = row["session_version"]
                raise SessionConcurrencyError(
                    f"Session concurrency conflict on {session.session_id}: "
                    f"expected version {expected_version}, but found {actual_version}"
                )
            self._conn.commit()
            session.session_version = new_version
            return session

    def _row_to_session(self, row: sqlite3.Row) -> AISession:
        return AISession(
            session_id=row["session_id"],
            conversation_id=row["conversation_id"],
            session_version=row["session_version"],
            session_state=json.loads(row["session_state"]),
            business_references=json.loads(row["business_references"]),
            interaction_results=json.loads(row["interaction_results"]),
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            closed_at=datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
        )

    # --- SessionHistoryRepository ---
    def append_message(self, message: SessionMessage) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_messages (
                    message_id, session_id, conversation_id, role, content, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    json.dumps(message.metadata),
                    message.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def get_messages_for_session(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[SessionMessage]:
        with self._lock:
            cur = self._conn.cursor()
            if limit:
                cur.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM session_messages
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ) ORDER BY created_at ASC
                    """,
                    (session_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM session_messages WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                )
            rows = cur.fetchall()
            return [
                SessionMessage(
                    message_id=r["message_id"],
                    session_id=r["session_id"],
                    conversation_id=r["conversation_id"],
                    role=r["role"],
                    content=r["content"],
                    metadata=json.loads(r["metadata"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    def get_messages_for_conversation(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[SessionMessage]:
        with self._lock:
            cur = self._conn.cursor()
            if limit:
                cur.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM session_messages
                        WHERE conversation_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ) ORDER BY created_at ASC
                    """,
                    (conversation_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM session_messages WHERE conversation_id = ? ORDER BY created_at ASC",
                    (conversation_id,),
                )
            rows = cur.fetchall()
            return [
                SessionMessage(
                    message_id=r["message_id"],
                    session_id=r["session_id"],
                    conversation_id=r["conversation_id"],
                    role=r["role"],
                    content=r["content"],
                    metadata=json.loads(r["metadata"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    def get_previous_sessions(
        self, conversation_id: str, current_session_id: str, limit: int = 10
    ) -> List[str]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT session_id FROM session_messages
                WHERE conversation_id = ? AND session_id != ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, current_session_id, limit),
            )
            rows = cur.fetchall()
            return [r["session_id"] for r in reversed(rows)]

    # --- OnlineTemplateRepository ---
    def get_template(self, case_key: str, locale: str) -> Optional[OnlineTemplate]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM online_templates WHERE case_key = ? AND locale = ?",
                (case_key, locale.lower()),
            )
            row = cur.fetchone()
            if not row:
                return None
            return OnlineTemplate(
                template_id=row["template_id"],
                case_key=row["case_key"],
                locale=row["locale"],
                template_text=row["template_text"],
                required_variables=json.loads(row["required_variables"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def save_template(self, template: OnlineTemplate) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO online_templates (
                    template_id, case_key, locale, template_text, required_variables, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_key, locale) DO UPDATE SET
                    template_text = excluded.template_text,
                    required_variables = excluded.required_variables
                """,
                (
                    template.template_id,
                    template.case_key,
                    template.locale.lower(),
                    template.template_text,
                    json.dumps(template.required_variables),
                    template.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def list_templates(self, case_key: Optional[str] = None) -> List[OnlineTemplate]:
        with self._lock:
            cur = self._conn.cursor()
            if case_key:
                cur.execute("SELECT * FROM online_templates WHERE case_key = ?", (case_key,))
            else:
                cur.execute("SELECT * FROM online_templates")
            rows = cur.fetchall()
            return [
                OnlineTemplate(
                    template_id=r["template_id"],
                    case_key=r["case_key"],
                    locale=r["locale"],
                    template_text=r["template_text"],
                    required_variables=json.loads(r["required_variables"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    # --- AuditRepository ---
    def record_audit(self, record: AuditRecord) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO audit_records (
                    audit_id, execution_id, session_id, conversation_id, case_key,
                    prompt_version_number, prompt_version_hash, endpoint_catalog_revision,
                    variable_catalog_revision, effective_config, inputs, outputs, telemetry, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    outputs = excluded.outputs,
                    telemetry = excluded.telemetry
                """,
                (
                    record.audit_id,
                    record.execution_id,
                    record.session_id,
                    record.conversation_id,
                    record.case_key,
                    record.prompt_version_number,
                    record.prompt_version_hash,
                    record.endpoint_catalog_revision,
                    record.variable_catalog_revision,
                    json.dumps(record.effective_configuration),
                    json.dumps(record.inputs),
                    json.dumps(record.outputs) if record.outputs is not None else None,
                    json.dumps(record.telemetry),
                    record.timestamp.isoformat(),
                ),
            )
            self._conn.commit()

    def get_by_execution_id(self, execution_id: str) -> Optional[AuditRecord]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM audit_records WHERE execution_id = ?", (execution_id,))
            row = cur.fetchone()
            if not row:
                return None
            return AuditRecord(
                audit_id=row["audit_id"],
                execution_id=row["execution_id"],
                session_id=row["session_id"],
                conversation_id=row["conversation_id"],
                case_key=row["case_key"],
                prompt_version_number=row["prompt_version_number"],
                prompt_version_hash=row["prompt_version_hash"],
                endpoint_catalog_revision=row["endpoint_catalog_revision"],
                variable_catalog_revision=row["variable_catalog_revision"],
                effective_configuration=json.loads(row["effective_config"]),
                inputs=json.loads(row["inputs"]),
                outputs=json.loads(row["outputs"]) if row["outputs"] else None,
                telemetry=json.loads(row["telemetry"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )

    def list_by_session_id(self, session_id: str) -> List[AuditRecord]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM audit_records WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
            rows = cur.fetchall()
            return [
                AuditRecord(
                    audit_id=r["audit_id"],
                    execution_id=r["execution_id"],
                    session_id=r["session_id"],
                    conversation_id=r["conversation_id"],
                    case_key=r["case_key"],
                    prompt_version_number=r["prompt_version_number"],
                    prompt_version_hash=r["prompt_version_hash"],
                    endpoint_catalog_revision=r["endpoint_catalog_revision"],
                    variable_catalog_revision=r["variable_catalog_revision"],
                    effective_configuration=json.loads(r["effective_config"]),
                    inputs=json.loads(r["inputs"]),
                    outputs=json.loads(r["outputs"]) if r["outputs"] else None,
                    telemetry=json.loads(r["telemetry"]),
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                for r in rows
            ]
