"""Storage contract.

The store owns durable state and transactions. Retrieval indexes are derived
state and must be rebuildable from stored records.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from memory_engine.models import (
    Agent,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    Project,
    ProjectMembership,
    Provenance,
    Session,
)


class MemoryStore(Protocol):
    """Durable storage operations used by the engine."""

    def add_agent(self, agent: Agent) -> Agent: ...

    def get_agent(self, agent_id: UUID) -> Agent | None: ...

    def add_project(self, project: Project) -> Project: ...

    def get_project(self, project_id: UUID) -> Project | None: ...

    def add_session(self, session: Session) -> Session: ...

    def get_session(self, session_id: UUID) -> Session | None: ...

    def add_membership(self, membership: ProjectMembership) -> ProjectMembership: ...

    def get_membership(
        self, project_id: UUID, agent_id: UUID
    ) -> ProjectMembership | None: ...

    def add_memory(self, memory: MemoryRecord) -> MemoryRecord: ...

    def get_memory(self, memory_id: UUID) -> MemoryRecord | None: ...

    def set_memory_status(
        self,
        memory_id: UUID,
        status: MemoryStatus,
        *,
        superseded_by_id: UUID | None = None,
        updated_at: datetime | None = None,
    ) -> bool: ...

    def list_visible_memories(
        self,
        *,
        agent_id: UUID,
        project_id: UUID | None = None,
        tiers: Iterable[MemoryTier] | None = None,
        limit: int = 50,
        at: datetime | None = None,
    ) -> list[MemoryRecord]: ...

    def get_history(self, memory_id: UUID) -> list[MemoryRecord]: ...

    def add_provenance(self, provenance: Provenance) -> Provenance: ...

    def list_provenance(self, memory_id: UUID) -> list[Provenance]: ...

    def close(self) -> None: ...
