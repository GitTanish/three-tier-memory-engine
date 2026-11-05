"""Tests for the core domain models."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memory_engine.models import (
    AccessScope,
    Agent,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    Project,
    ProjectMembership,
    ProjectRole,
    Provenance,
    ProvenanceRelation,
    Session,
)


def make_memory(**overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "tier": MemoryTier.SEMANTIC,
        "content": "The project uses PostgreSQL.",
        "owner_agent_id": uuid4(),
    }
    values.update(overrides)
    return MemoryRecord(**values)


def test_memory_defaults() -> None:
    memory = make_memory()

    assert memory.status == MemoryStatus.ACTIVE
    assert memory.scope == AccessScope.PRIVATE
    assert memory.version == 1
    assert memory.importance == 0.5
    assert memory.confidence == 1.0
    assert memory.tags == []
    assert memory.metadata == {}


def test_memory_content_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        make_memory(content="")


def test_project_scope_requires_project_id() -> None:
    with pytest.raises(ValidationError):
        make_memory(scope=AccessScope.PROJECT)


def test_project_scope_accepts_project_id() -> None:
    project_id = uuid4()
    memory = make_memory(scope=AccessScope.PROJECT, project_id=project_id)

    assert memory.project_id == project_id


def test_correction_chain_keeps_old_version() -> None:
    old = make_memory()
    new = make_memory(
        content="The project uses SQLite.",
        version=2,
        supersedes_id=old.id,
    )
    old = old.model_copy(
        update={
            "status": MemoryStatus.SUPERSEDED,
            "superseded_by_id": new.id,
        }
    )

    assert new.supersedes_id == old.id
    assert old.superseded_by_id == new.id
    assert old.status == MemoryStatus.SUPERSEDED
    assert new.status == MemoryStatus.ACTIVE


def test_working_memory_expiration() -> None:
    expired = make_memory(
        tier=MemoryTier.WORKING,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    current = make_memory(
        tier=MemoryTier.WORKING,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    assert expired.is_expired()
    assert not current.is_expired()
    assert not make_memory().is_expired()


def test_session_rejects_end_before_start() -> None:
    started_at = datetime.now(UTC)

    with pytest.raises(ValidationError):
        Session(
            agent_id=uuid4(),
            project_id=uuid4(),
            started_at=started_at,
            ended_at=started_at - timedelta(seconds=1),
        )


def test_session_accepts_valid_end() -> None:
    started_at = datetime.now(UTC)
    session = Session(
        agent_id=uuid4(),
        project_id=uuid4(),
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=5),
    )

    assert session.ended_at is not None
    assert session.ended_at > session.started_at


def test_provenance_requires_a_source() -> None:
    with pytest.raises(ValidationError):
        Provenance(
            memory_id=uuid4(),
            relation=ProvenanceRelation.DERIVED_FROM,
        )


def test_provenance_accepts_memory_source() -> None:
    source_id = uuid4()
    provenance = Provenance(
        memory_id=uuid4(),
        relation=ProvenanceRelation.DERIVED_FROM,
        source_memory_id=source_id,
    )

    assert provenance.source_memory_id == source_id


def test_project_membership_defaults_to_reader() -> None:
    membership = ProjectMembership(project_id=uuid4(), agent_id=uuid4())

    assert membership.role == ProjectRole.READER


def test_postgres_to_sqlite_correction_example() -> None:
    agent_a = Agent(name="Agent A")
    agent_b = Agent(name="Agent B")
    project = Project(name="Demo project")

    old = make_memory(
        owner_agent_id=agent_a.id,
        project_id=project.id,
        scope=AccessScope.PROJECT,
    )
    new = make_memory(
        content="The project uses SQLite.",
        owner_agent_id=agent_a.id,
        project_id=project.id,
        scope=AccessScope.PROJECT,
        version=2,
        supersedes_id=old.id,
    )
    old = old.model_copy(
        update={
            "status": MemoryStatus.SUPERSEDED,
            "superseded_by_id": new.id,
        }
    )
    correction = Provenance(
        memory_id=new.id,
        relation=ProvenanceRelation.CORRECTED_BY,
        source_memory_id=old.id,
        source_agent_id=agent_a.id,
    )
    membership = ProjectMembership(
        project_id=project.id,
        agent_id=agent_b.id,
        role=ProjectRole.READER,
    )

    assert new.content == "The project uses SQLite."
    assert old.content == "The project uses PostgreSQL."
    assert old.status == MemoryStatus.SUPERSEDED
    assert correction.source_memory_id == old.id
    assert membership.project_id == project.id

