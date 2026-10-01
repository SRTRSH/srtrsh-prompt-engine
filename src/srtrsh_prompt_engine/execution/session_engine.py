"""AI Session lifecycle management and history retrieval.

Normative references:
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/01-session-model.md
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/02-session-history.md
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/03-session-state.md
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/04-session-continuation.md
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/06-session-concurrency.md
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from srtrsh_prompt_engine.domain.models import (
    AISession,
    SessionMessage,
)
from srtrsh_prompt_engine.storage.repository import (
    AISessionRepository,
    SessionHistoryRepository,
)


class SessionEngine:
    """Manages AI Session continuity, closure, creation, and durable history selection."""

    def __init__(
        self,
        session_repository: AISessionRepository,
        history_repository: SessionHistoryRepository,
        default_continuation_window_seconds: int = 1800,  # 30 minutes
    ) -> None:
        self.session_repo = session_repository
        self.history_repo = history_repository
        self.default_continuation_window_seconds = default_continuation_window_seconds

    def get_or_create_session(
        self,
        conversation_id: str,
        continuation_window_seconds: Optional[int] = None,
        business_references: Optional[Dict[str, Any]] = None,
    ) -> AISession:
        """Resolve active session or create a new session if outside continuation window (C-25).

        Exceeding continuation window closes the active session and creates a new session,
        but does NOT delete or truncate retained historical conversation.
        """
        window_sec = (
            continuation_window_seconds
            if continuation_window_seconds is not None
            else self.default_continuation_window_seconds
        )
        now = datetime.now(timezone.utc)

        active = self.session_repo.get_active_by_conversation_id(conversation_id)
        if active:
            # Check elapsed time since last activity
            elapsed = (now - active.updated_at).total_seconds()
            if elapsed <= window_sec:
                if business_references:
                    active.business_references.update(business_references)
                return active
            else:
                # Outside continuation window: close existing session
                active.is_active = False
                active.closed_at = now
                self.session_repo.save(active)

        # Create new session initialized to session_version=1
        new_session = AISession(
            session_id=f"sess_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            session_version=1,
            session_state={},
            business_references=business_references or {},
            interaction_results=[],
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self.session_repo.save(new_session)
        return new_session

    def select_history_context(
        self,
        conversation_id: str,
        current_session_id: str,
        recent_messages_limit: int = 10,
        previous_sessions_limit: int = 2,
    ) -> List[SessionMessage]:
        """Select bounded history for LLM context (C-25, C-26):

        recentMessages = N + previousSessions = M.
        Durable persistence is not altered by this selection view.
        """
        combined: List[SessionMessage] = []

        # 1. Fetch previous sessions context if configured
        if previous_sessions_limit > 0:
            prev_session_ids = self.history_repo.get_previous_sessions(
                conversation_id, current_session_id, limit=previous_sessions_limit
            )
            for prev_sid in prev_session_ids:
                prev_msgs = self.history_repo.get_messages_for_session(prev_sid, limit=5)
                combined.extend(prev_msgs)

        # 2. Fetch current session messages up to recent_messages_limit
        curr_msgs = self.history_repo.get_messages_for_session(
            current_session_id, limit=recent_messages_limit
        )
        combined.extend(curr_msgs)
        return combined

    def record_message(
        self,
        session_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionMessage:
        """Durably persist conversation message (C-26)."""
        msg = SessionMessage(
            session_id=session_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.history_repo.append_message(msg)
        return msg

    def update_session(self, session: AISession, expected_version: int) -> AISession:
        """Atomically advance session version with optimistic locking (C-17)."""
        return self.session_repo.update_with_optimistic_lock(session, expected_version)
