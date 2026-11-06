"""SQLite-backed storage.

SQLite is the source of truth. Retrieval indexes are derived state and can be
rebuilt from these records.
"""

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from memory_engine.models import (
    AccessScope,
    Agent,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    Project,
    ProjectMembership,
    Provenance,
    ProvenanceRelation,
    Session,
)
from memory_engine.models.base import utc_now

SCHEMA_PATH = Path(__file__).parent / "migrations" / "0001_initial.sql"


def _dt_to_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _uuid_to_text(value: UUID | None) -> str | None:
    return str(value) if value is not None else None


class SqliteMemoryStore:
    """SQLite implementation of the storage contract."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SqliteMemoryStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def add_agent(self, agent: Agent) -> Agent:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO agents (id, name, created_at, is_active)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(agent.id),
                    agent.name,
                    _dt_to_text(agent.created_at),
                    int(agent.is_active),
                ),
            )
        return agent

    def get_agent(self, agent_id: UUID) -> Agent | None:
        row = self._connection.execute(
            "SELECT * FROM agents WHERE id = ?", (str(agent_id),)
        ).fetchone()
        if row is None:
            return None
        return Agent(
            id=UUID(row["id"]),
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            is_active=bool(row["is_active"]),
        )

    def add_project(self, project: Project) -> Project:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO projects (id, name, created_at)
                VALUES (?, ?, ?)
                """,
                (str(project.id), project.name, _dt_to_text(project.created_at)),
            )
        return project

    def get_project(self, project_id: UUID) -> Project | None:
        row = self._connection.execute(
            "SELECT * FROM projects WHERE id = ?", (str(project_id),)
        ).fetchone()
        if row is None:
            return None
        return Project(
            id=UUID(row["id"]),
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def add_session(self, session: Session) -> Session:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions (
                    id, agent_id, project_id, task_id, started_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(session.id),
                    str(session.agent_id),
                    str(session.project_id),
                    session.task_id,
                    _dt_to_text(session.started_at),
                    _dt_to_text(session.ended_at),
                ),
            )
        return session

    def get_session(self, session_id: UUID) -> Session | None:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (str(session_id),)
        ).fetchone()
        if row is None:
            return None
        return Session(
            id=UUID(row["id"]),
            agent_id=UUID(row["agent_id"]),
            project_id=UUID(row["project_id"]),
            task_id=row["task_id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(
                datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
            ),
        )

    def add_membership(self, membership: ProjectMembership) -> ProjectMembership:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO project_memberships (
                    project_id, agent_id, role, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(membership.project_id),
                    str(membership.agent_id),
                    membership.role.value,
                    _dt_to_text(membership.created_at),
                ),
            )
        return membership

    def get_membership(
        self, project_id: UUID, agent_id: UUID
    ) -> ProjectMembership | None:
        row = self._connection.execute(
            """
            SELECT * FROM project_memberships
            WHERE project_id = ? AND agent_id = ?
            """,
            (str(project_id), str(agent_id)),
        ).fetchone()
        if row is None:
            return None
        return ProjectMembership(
            project_id=UUID(row["project_id"]),
            agent_id=UUID(row["agent_id"]),
            role=row["role"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def add_memory(self, memory: MemoryRecord) -> MemoryRecord:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO memories (
                    id, tier, content, summary, owner_agent_id, project_id,
                    session_id, scope, importance, confidence, status, version,
                    supersedes_id, superseded_by_id, created_at, updated_at,
                    expires_at, last_accessed_at, tags_json, metadata_json
                ) VALUES (
                    :id, :tier, :content, :summary, :owner_agent_id, :project_id,
                    :session_id, :scope, :importance, :confidence, :status, :version,
                    :supersedes_id, :superseded_by_id, :created_at, :updated_at,
                    :expires_at, :last_accessed_at, :tags_json, :metadata_json
                )
                """,
                {
                    "id": str(memory.id),
                    "tier": memory.tier.value,
                    "content": memory.content,
                    "summary": memory.summary,
                    "owner_agent_id": str(memory.owner_agent_id),
                    "project_id": _uuid_to_text(memory.project_id),
                    "session_id": _uuid_to_text(memory.session_id),
                    "scope": memory.scope.value,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "status": memory.status.value,
                    "version": memory.version,
                    "supersedes_id": _uuid_to_text(memory.supersedes_id),
                    "superseded_by_id": _uuid_to_text(memory.superseded_by_id),
                    "created_at": _dt_to_text(memory.created_at),
                    "updated_at": _dt_to_text(memory.updated_at),
                    "expires_at": _dt_to_text(memory.expires_at),
                    "last_accessed_at": _dt_to_text(memory.last_accessed_at),
                    "tags_json": json.dumps(memory.tags),
                    "metadata_json": json.dumps(memory.metadata, default=str),
                },
            )
        return memory

    def get_memory(self, memory_id: UUID) -> MemoryRecord | None:
        row = self._connection.execute(
            "SELECT * FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
        return self._row_to_memory(row) if row is not None else None

    def set_memory_status(
        self,
        memory_id: UUID,
        status: MemoryStatus,
        *,
        superseded_by_id: UUID | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE memories
                SET status = ?,
                    superseded_by_id = COALESCE(?, superseded_by_id),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    _uuid_to_text(superseded_by_id),
                    _dt_to_text(updated_at or utc_now()),
                    str(memory_id),
                ),
            )
        return cursor.rowcount > 0

    def list_visible_memories(
        self,
        *,
        agent_id: UUID,
        project_id: UUID | None = None,
        tiers: Iterable[MemoryTier] | None = None,
        limit: int = 50,
        at: datetime | None = None,
    ) -> list[MemoryRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        tier_values = [tier.value for tier in tiers] if tiers is not None else []
        if tiers is not None and not tier_values:
            return []

        conditions = [
            "m.status = ?",
            "(m.expires_at IS NULL OR m.expires_at > ?)",
            """
            (
                m.owner_agent_id = ?
                OR (
                    m.scope = ?
                    AND m.project_id IN (
                        SELECT project_id FROM project_memberships WHERE agent_id = ?
                    )
                )
            )
            """,
        ]
        params: list[object] = [
            MemoryStatus.ACTIVE.value,
            _dt_to_text(at or utc_now()),
            str(agent_id),
            AccessScope.PROJECT.value,
            str(agent_id),
        ]

        if project_id is not None:
            conditions.append("m.project_id = ?")
            params.append(str(project_id))

        if tier_values:
            placeholders = ", ".join("?" for _ in tier_values)
            conditions.append(f"m.tier IN ({placeholders})")
            params.extend(tier_values)

        params.append(limit)
        rows = self._connection.execute(
            f"""
            SELECT m.* FROM memories m
            WHERE {" AND ".join(conditions)}
            ORDER BY m.importance DESC, m.created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_history(self, memory_id: UUID) -> list[MemoryRecord]:
        current = self.get_memory(memory_id)
        if current is None:
            return []

        seen: set[UUID] = set()
        while current.supersedes_id is not None and current.supersedes_id not in seen:
            seen.add(current.id)
            previous = self.get_memory(current.supersedes_id)
            if previous is None:
                break
            current = previous

        chain: list[MemoryRecord] = []
        seen.clear()
        while current is not None and current.id not in seen:
            seen.add(current.id)
            chain.append(current)
            if current.superseded_by_id is None:
                break
            current = self.get_memory(current.superseded_by_id)
        return chain

    def add_provenance(self, provenance: Provenance) -> Provenance:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO provenance (
                    id, memory_id, relation, source_memory_id,
                    source_session_id, source_agent_id, source_excerpt, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(provenance.id),
                    str(provenance.memory_id),
                    provenance.relation.value,
                    _uuid_to_text(provenance.source_memory_id),
                    _uuid_to_text(provenance.source_session_id),
                    _uuid_to_text(provenance.source_agent_id),
                    provenance.source_excerpt,
                    _dt_to_text(provenance.created_at),
                ),
            )
        return provenance

    def list_provenance(self, memory_id: UUID) -> list[Provenance]:
        rows = self._connection.execute(
            """
            SELECT * FROM provenance
            WHERE memory_id = ?
            ORDER BY created_at ASC
            """,
            (str(memory_id),),
        ).fetchall()
        return [
            Provenance(
                id=UUID(row["id"]),
                memory_id=UUID(row["memory_id"]),
                relation=ProvenanceRelation(row["relation"]),
                source_memory_id=(
                    UUID(row["source_memory_id"]) if row["source_memory_id"] else None
                ),
                source_session_id=(
                    UUID(row["source_session_id"])
                    if row["source_session_id"]
                    else None
                ),
                source_agent_id=(
                    UUID(row["source_agent_id"]) if row["source_agent_id"] else None
                ),
                source_excerpt=row["source_excerpt"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=UUID(row["id"]),
            tier=MemoryTier(row["tier"]),
            content=row["content"],
            summary=row["summary"],
            owner_agent_id=UUID(row["owner_agent_id"]),
            project_id=UUID(row["project_id"]) if row["project_id"] else None,
            session_id=UUID(row["session_id"]) if row["session_id"] else None,
            scope=AccessScope(row["scope"]),
            importance=row["importance"],
            confidence=row["confidence"],
            status=MemoryStatus(row["status"]),
            version=row["version"],
            supersedes_id=(
                UUID(row["supersedes_id"]) if row["supersedes_id"] else None
            ),
            superseded_by_id=(
                UUID(row["superseded_by_id"]) if row["superseded_by_id"] else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=(
                datetime.fromisoformat(row["expires_at"])
                if row["expires_at"]
                else None
            ),
            last_accessed_at=(
                datetime.fromisoformat(row["last_accessed_at"])
                if row["last_accessed_at"]
                else None
            ),
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
        )
