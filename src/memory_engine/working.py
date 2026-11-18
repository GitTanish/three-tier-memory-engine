"""Session-scoped working scratchpad."""

from datetime import timedelta
from typing import Any
from uuid import UUID

from memory_engine.errors import AccessDeniedError, NotFoundError
from memory_engine.models import Session, WorkingEntry
from memory_engine.models.base import utc_now
from memory_engine.storage.base import MemoryStore

DEFAULT_TTL_SECONDS = 3600


class WorkingScratchpad:
    """Key/value state for the current task.

    A scratchpad belongs to one session and one agent. Entries expire. Reads
    hide expired entries, and ``evict_expired`` removes them.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def set(
        self,
        *,
        agent_id: UUID,
        session_id: UUID,
        key: str,
        value: Any,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        importance: float = 0.5,
    ) -> WorkingEntry:
        """Create or replace a scratchpad entry."""
        self._require_session(agent_id, session_id)
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")

        now = utc_now()
        entry = WorkingEntry(
            session_id=session_id,
            key=key,
            value=value,
            importance=importance,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        try:
            self._store.upsert_working_entry(entry)
        except TypeError as exc:
            raise ValueError("working entry value must be JSON-serializable") from exc

        stored = self._store.get_working_entry(session_id, key)
        if stored is None:
            raise RuntimeError("working entry was not persisted")
        return stored

    def get(
        self,
        *,
        agent_id: UUID,
        session_id: UUID,
        key: str,
    ) -> WorkingEntry | None:
        """Return an active entry, or ``None`` if it is missing or expired."""
        self._require_session(agent_id, session_id)
        return self._store.get_working_entry(session_id, key)

    def list(self, *, agent_id: UUID, session_id: UUID) -> list[WorkingEntry]:
        """Return active entries for a session, newest first."""
        self._require_session(agent_id, session_id)
        return self._store.list_working_entries(session_id)

    def delete(self, *, agent_id: UUID, session_id: UUID, key: str) -> bool:
        """Delete one entry. Returns whether it existed."""
        self._require_session(agent_id, session_id)
        return self._store.delete_working_entry(session_id, key)

    def clear(self, *, agent_id: UUID, session_id: UUID) -> int:
        """Delete every entry in a session. Returns the number removed."""
        self._require_session(agent_id, session_id)
        return self._store.clear_working_entries(session_id)

    def evict_expired(self) -> int:
        """Remove every expired entry across all sessions."""
        return self._store.evict_expired_working_entries()

    def _require_session(self, agent_id: UUID, session_id: UUID) -> Session:
        agent = self._store.get_agent(agent_id)
        if agent is None:
            raise NotFoundError(f"agent {agent_id} not found")
        if not agent.is_active:
            raise AccessDeniedError(f"agent {agent_id} is inactive")

        session = self._store.get_session(session_id)
        if session is None:
            raise NotFoundError(f"session {session_id} not found")
        if session.agent_id != agent_id:
            raise AccessDeniedError("agent cannot access this session")
        return session
