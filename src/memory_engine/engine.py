"""Public MemoryEngine facade.

The engine coordinates storage, access control, and correction semantics
behind one interface. Transport and retrieval ranking are added separately.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import NamedTuple
from uuid import UUID

from memory_engine.compression import build_session_episode
from memory_engine.embeddings import Embedder, HashingEmbedder
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
    RecallHit,
    Session,
)
from memory_engine.models.base import utc_now
from memory_engine.retrieval import apply_token_budget, rank_memories
from memory_engine.storage.base import MemoryStore
from memory_engine.working import WorkingScratchpad


class EvictionResult(NamedTuple):
    """Counts of rows removed or expired by one sweep."""

    working_entries: int
    memories: int


class MemoryEngine:
    """Coordinates memory storage, access checks, corrections, and recall."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder or HashingEmbedder()
        self.working = WorkingScratchpad(store)

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

    def start_session(
        self,
        *,
        agent_id: UUID,
        project_id: UUID,
        task_id: str | None = None,
    ) -> Session:
        self._require_agent(agent_id)
        if self._store.get_project(project_id) is None:
            raise NotFoundError(f"project {project_id} not found")
        if self._store.get_membership(project_id, agent_id) is None:
            raise AccessDeniedError("agent is not a project member")
        return self._store.add_session(
            Session(agent_id=agent_id, project_id=project_id, task_id=task_id)
        )

    def end_session(
        self,
        *,
        agent_id: UUID,
        session_id: UUID,
        at: datetime | None = None,
        compress: bool = False,
    ) -> Session:
        """Close a session and clear its scratch state.

        ``compress=True`` creates the session episode after stamping the end
        time and before clearing working state, so the episode records the
        real end. It stays off by default because episodes are project-scoped
        and may share private working state. Calling this again returns the
        already closed session unchanged.
        """
        self._require_agent(agent_id)
        session = self._store.get_session(session_id)
        if session is None:
            raise NotFoundError(f"session {session_id} not found")
        if session.agent_id != agent_id:
            raise AccessDeniedError("agent cannot end this session")
        if session.ended_at is not None:
            return session

        ended_at = at or utc_now()
        if ended_at < session.started_at:
            raise ValueError("ended_at must not be earlier than started_at")

        self._store.set_session_ended(session_id, ended_at)
        if compress and self._session_has_content(agent_id, session_id):
            self.compress_session(agent_id=agent_id, session_id=session_id)
        self.working.clear(agent_id=agent_id, session_id=session_id)
        updated = self._store.get_session(session_id)
        if updated is None:
            raise RuntimeError("session was not persisted")
        return updated

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

        stored = self._store.add_memory(memory)
        self._index_memory(stored)
        return stored

    def list_memories(
        self,
        *,
        agent_id: UUID,
        project_id: UUID | None = None,
        session_id: UUID | None = None,
        tiers: Iterable[MemoryTier] | None = None,
        limit: int | None = 50,
    ) -> list[MemoryRecord]:
        self._require_agent(agent_id)
        return self._store.list_visible_memories(
            agent_id=agent_id,
            project_id=project_id,
            session_id=session_id,
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

    def recall(
        self,
        *,
        agent_id: UUID,
        query: str,
        project_id: UUID | None = None,
        session_id: UUID | None = None,
        tiers: Iterable[MemoryTier] | None = (MemoryTier.SEMANTIC,),
        limit: int = 10,
        max_tokens: int | None = None,
    ) -> list[RecallHit]:
        """Rank visible memories for a query, best first.

        Access filtering happens in storage before ranking. Superseded and
        expired memories are excluded there too. Missing embeddings are
        rebuilt lazily, so recall stays correct after a restore. Each
        returned hit counts as one access, which feeds importance scoring.
        """
        self._require_agent(agent_id)
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        # shortcut: ranks every visible candidate; add a candidate prefilter
        # when memory counts outgrow a local scan.
        candidates = self._store.list_visible_memories(
            agent_id=agent_id,
            project_id=project_id,
            session_id=session_id,
            tiers=tiers,
            limit=None,
        )
        stored = self._store.get_memory_embeddings(
            memory.id for memory in candidates
        )
        embeddings: dict[UUID, list[float]] = {}
        for memory in candidates:
            entry = stored.get(memory.id)
            if entry is None or entry[0] != self._embedder.id:
                self._index_memory(memory)
                entry = (self._embedder.id, self._embedder.embed(memory.content))
            embeddings[memory.id] = entry[1]

        hits = rank_memories(
            query=query,
            memories=candidates,
            embeddings=embeddings,
            embedder=self._embedder,
        )
        selected = apply_token_budget(hits, max_tokens)[:limit]
        self._store.record_memory_access(hit.memory.id for hit in selected)
        return selected

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
        self._index_memory(new)

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

    def compress_session(
        self,
        *,
        agent_id: UUID,
        session_id: UUID,
        importance: float | None = None,
    ) -> MemoryRecord:
        """Compress one session into a deterministic episodic memory.

        The application decides when a session is ready. Calling this again
        returns the existing episode. Working entries and source memories are
        left in place so the caller can clear or expire them separately.

        The episode is project-scoped, so compression shares the session
        summary with project members.
        """
        agent = self._require_agent(agent_id)
        session = self._store.get_session(session_id)
        if session is None:
            raise NotFoundError(f"session {session_id} not found")
        if session.agent_id != agent_id:
            raise AccessDeniedError("agent cannot compress this session")

        project = self._store.get_project(session.project_id)
        if project is None:
            raise NotFoundError(f"project {session.project_id} not found")

        episodic_memories = self._store.list_visible_memories(
            agent_id=agent_id,
            project_id=project.id,
            session_id=session.id,
            tiers=[MemoryTier.EPISODIC],
            limit=None,
        )
        existing = max(
            (
                memory
                for memory in episodic_memories
                if memory.metadata.get("kind") == "session_episode"
            ),
            key=lambda memory: memory.created_at,
            default=None,
        )
        if existing is not None:
            return existing

        working_entries = self._store.list_working_entries(session.id)
        if not working_entries and not episodic_memories:
            raise InvalidStateError("session has nothing to compress")

        episode, provenance = build_session_episode(
            session=session,
            agent=agent,
            project=project,
            working_entries=working_entries,
            episodic_memories=episodic_memories,
            importance=importance,
        )
        stored = self.store_memory(episode)
        for edge in provenance:
            self._store.add_provenance(edge)
        return stored

    def evict_expired(self, *, at: datetime | None = None) -> EvictionResult:
        """Remove expired working entries and expire outdated memories.

        Working memory is deleted because it is scratch state. Episodic and
        semantic memories are marked expired, so their history and provenance
        remain available. Only active memories change status.
        """
        return EvictionResult(
            working_entries=self.working.evict_expired(at=at),
            memories=self._store.expire_memories(at=at),
        )

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

    def _index_memory(self, memory: MemoryRecord) -> None:
        self._store.upsert_memory_embedding(
            memory.id,
            self._embedder.id,
            self._embedder.embed(memory.content),
        )

    def _session_has_content(self, agent_id: UUID, session_id: UUID) -> bool:
        if self._store.list_working_entries(session_id):
            return True
        return bool(
            self._store.list_visible_memories(
                agent_id=agent_id,
                session_id=session_id,
                tiers=[MemoryTier.EPISODIC],
                limit=1,
            )
        )

    def _require_agent(self, agent_id: UUID) -> Agent:
        agent = self._store.get_agent(agent_id)
        if agent is None:
            raise NotFoundError(f"agent {agent_id} not found")
        if not agent.is_active:
            raise AccessDeniedError(f"agent {agent_id} is inactive")
        return agent
