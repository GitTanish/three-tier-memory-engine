"""Public MemoryEngine facade.

The engine coordinates storage, access control, and correction semantics
behind one interface. Transport and retrieval ranking are added separately.
"""

from collections.abc import Iterable
from uuid import UUID

from memory_engine.errors import AccessDeniedError, InvalidStateError, NotFoundError
from memory_engine.models import (
    AccessScope,
    Agent,
    CorrectionKind,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    Project,
    ProjectMembership,
    ProjectRole,
    Provenance,
    ProvenanceRelation,
)
from memory_engine.models.base import utc_now
from memory_engine.storage.base import MemoryStore


class MemoryEngine:
    """Coordinates memory storage, access checks, and corrections."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def register_agent(self, name: str) -> Agent:
        return self._store.add_agent(Agent(name=name))

    def create_project(self, name: str) -> Project:
        return self._store.add_project(Project(name=name))

    def add_project_member(
        self,
        project_id: UUID,
        agent_id: UUID,
        role: ProjectRole = ProjectRole.READER,
    ) -> ProjectMembership:
        self._require_agent(agent_id)
        if self._store.get_project(project_id) is None:
            raise NotFoundError(f"project {project_id} not found")
        if self._store.get_membership(project_id, agent_id) is not None:
            raise InvalidStateError("agent is already a project member")
        return self._store.add_membership(
            ProjectMembership(project_id=project_id, agent_id=agent_id, role=role)
        )

    def store_memory(self, memory: MemoryRecord) -> MemoryRecord:
        self._require_agent(memory.owner_agent_id)

        if (
            memory.project_id is not None
            and self._store.get_project(memory.project_id) is None
        ):
            raise NotFoundError(f"project {memory.project_id} not found")
        if (
            memory.session_id is not None
            and self._store.get_session(memory.session_id) is None
        ):
            raise NotFoundError(f"session {memory.session_id} not found")

        if memory.scope == AccessScope.PROJECT:
            if memory.project_id is None:
                raise InvalidStateError("project-scoped memory requires a project_id")
            membership = self._store.get_membership(
                memory.project_id, memory.owner_agent_id
            )
            if membership is None or membership.role == ProjectRole.READER:
                raise AccessDeniedError("agent cannot write project-scoped memory")

        return self._store.add_memory(memory)

    def list_memories(
        self,
        *,
        agent_id: UUID,
        project_id: UUID | None = None,
        tiers: Iterable[MemoryTier] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        self._require_agent(agent_id)
        return self._store.list_visible_memories(
            agent_id=agent_id,
            project_id=project_id,
            tiers=tiers,
            limit=limit,
        )

    def get_memory(self, *, agent_id: UUID, memory_id: UUID) -> MemoryRecord:
        self._require_agent(agent_id)
        memory = self._store.get_memory(memory_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found")
        if not self.can_read(agent_id, memory):
            raise AccessDeniedError("agent cannot read this memory")
        return memory

    def get_history(self, *, agent_id: UUID, memory_id: UUID) -> list[MemoryRecord]:
        self._require_agent(agent_id)
        history = self._store.get_history(memory_id)
        if not history:
            raise NotFoundError(f"memory {memory_id} not found")
        visible = [memory for memory in history if self.can_read(agent_id, memory)]
        if not visible:
            raise AccessDeniedError("agent cannot read this memory")
        return visible

    def correct_memory(
        self,
        memory_id: UUID,
        *,
        actor_agent_id: UUID,
        content: str,
        kind: CorrectionKind = CorrectionKind.SUPERSEDED,
        reason: str | None = None,
    ) -> MemoryRecord:
        self._require_agent(actor_agent_id)
        old = self._store.get_memory(memory_id)
        if old is None:
            raise NotFoundError(f"memory {memory_id} not found")
        if not self.can_write(actor_agent_id, old):
            raise AccessDeniedError("agent cannot correct this memory")
        if old.status != MemoryStatus.ACTIVE:
            raise InvalidStateError("only active memories can be corrected")
        if old.is_expired():
            raise InvalidStateError("expired memories cannot be corrected")

        now = utc_now()
        new = MemoryRecord(
            tier=old.tier,
            content=content,
            summary=old.summary,
            owner_agent_id=old.owner_agent_id,
            project_id=old.project_id,
            session_id=old.session_id,
            scope=old.scope,
            importance=old.importance,
            confidence=old.confidence,
            status=MemoryStatus.ACTIVE,
            version=old.version + 1,
            supersedes_id=old.id,
            created_at=now,
            updated_at=now,
            expires_at=old.expires_at,
            tags=list(old.tags),
            metadata=dict(old.metadata),
        )
        self._store.add_memory(new)

        old_status = (
            MemoryStatus.SUPERSEDED
            if kind == CorrectionKind.SUPERSEDED
            else MemoryStatus.RETRACTED
        )
        self._store.set_memory_status(
            old.id,
            old_status,
            superseded_by_id=new.id,
            updated_at=now,
        )
        self._store.add_provenance(
            Provenance(
                memory_id=new.id,
                relation=ProvenanceRelation.CORRECTED_BY,
                source_memory_id=old.id,
                source_agent_id=actor_agent_id,
                source_excerpt=reason or old.content,
            )
        )
        return new

    def can_read(self, agent_id: UUID, memory: MemoryRecord) -> bool:
        if memory.owner_agent_id == agent_id:
            return True
        if memory.scope != AccessScope.PROJECT or memory.project_id is None:
            return False
        return self._store.get_membership(memory.project_id, agent_id) is not None

    def can_write(self, agent_id: UUID, memory: MemoryRecord) -> bool:
        if memory.owner_agent_id == agent_id:
            return True
        if memory.scope != AccessScope.PROJECT or memory.project_id is None:
            return False
        membership = self._store.get_membership(memory.project_id, agent_id)
        return membership is not None and membership.role in {
            ProjectRole.WRITER,
            ProjectRole.ADMIN,
        }

    def _require_agent(self, agent_id: UUID) -> Agent:
        agent = self._store.get_agent(agent_id)
        if agent is None:
            raise NotFoundError(f"agent {agent_id} not found")
        if not agent.is_active:
            raise AccessDeniedError(f"agent {agent_id} is inactive")
        return agent
