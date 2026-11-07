"""Integration tests for the MemoryEngine facade."""

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest

from memory_engine.engine import MemoryEngine
from memory_engine.errors import AccessDeniedError, InvalidStateError, NotFoundError
from memory_engine.models import (
    AccessScope,
    CorrectionKind,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    ProjectRole,
    ProvenanceRelation,
)
from memory_engine.storage import SqliteMemoryStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store() -> Iterator[SqliteMemoryStore]:
    database = SqliteMemoryStore()
    yield database
    database.close()


@pytest.fixture
def engine(store: SqliteMemoryStore) -> MemoryEngine:
    return MemoryEngine(store)


def store_memory(
    engine: MemoryEngine,
    *,
    owner_id: UUID,
    content: str = "The project uses PostgreSQL.",
    tier: MemoryTier = MemoryTier.SEMANTIC,
    **overrides: object,
) -> MemoryRecord:
    return engine.store_memory(
        MemoryRecord(
            tier=tier,
            content=content,
            owner_agent_id=owner_id,
            **overrides,
        )
    )


def test_postgres_to_sqlite_acceptance_flow(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent_a = engine.register_agent("Agent A")
    agent_b = engine.register_agent("Agent B")
    agent_c = engine.register_agent("Agent C")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, agent_a.id, ProjectRole.WRITER)
    engine.add_project_member(project.id, agent_b.id, ProjectRole.READER)

    old = store_memory(
        engine,
        owner_id=agent_a.id,
        project_id=project.id,
        scope=AccessScope.PROJECT,
    )

    assert engine.list_memories(agent_id=agent_b.id) == [old]

    new = engine.correct_memory(
        old.id,
        actor_agent_id=agent_a.id,
        content="The project uses SQLite.",
        reason="The initial setup note was wrong.",
    )

    visible = engine.list_memories(agent_id=agent_b.id)
    history = engine.get_history(agent_id=agent_b.id, memory_id=new.id)
    provenance = store.list_provenance(new.id)

    assert [memory.content for memory in visible] == ["The project uses SQLite."]
    assert [memory.content for memory in history] == [
        "The project uses PostgreSQL.",
        "The project uses SQLite.",
    ]
    assert history[0].status == MemoryStatus.SUPERSEDED
    assert history[1].status == MemoryStatus.ACTIVE
    assert engine.list_memories(agent_id=agent_c.id) == []
    assert len(provenance) == 1
    assert provenance[0].relation == ProvenanceRelation.CORRECTED_BY
    assert provenance[0].source_memory_id == old.id
    assert provenance[0].source_agent_id == agent_a.id


def test_private_memory_is_hidden_from_other_agents(
    engine: MemoryEngine,
) -> None:
    owner = engine.register_agent("Agent A")
    other = engine.register_agent("Agent B")
    memory = store_memory(engine, owner_id=owner.id)

    assert engine.list_memories(agent_id=owner.id) == [memory]
    assert engine.list_memories(agent_id=other.id) == []
    with pytest.raises(AccessDeniedError):
        engine.get_memory(agent_id=other.id, memory_id=memory.id)


def test_project_reader_cannot_correct_memory(
    engine: MemoryEngine,
) -> None:
    owner = engine.register_agent("Agent A")
    reader = engine.register_agent("Agent B")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, owner.id, ProjectRole.WRITER)
    engine.add_project_member(project.id, reader.id, ProjectRole.READER)
    memory = store_memory(
        engine,
        owner_id=owner.id,
        project_id=project.id,
        scope=AccessScope.PROJECT,
    )

    with pytest.raises(AccessDeniedError):
        engine.correct_memory(
            memory.id,
            actor_agent_id=reader.id,
            content="The project uses SQLite.",
        )


def test_project_writer_can_correct_memory(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    owner = engine.register_agent("Agent A")
    writer = engine.register_agent("Agent B")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, owner.id, ProjectRole.WRITER)
    engine.add_project_member(project.id, writer.id, ProjectRole.WRITER)
    memory = store_memory(
        engine,
        owner_id=owner.id,
        project_id=project.id,
        scope=AccessScope.PROJECT,
    )

    corrected = engine.correct_memory(
        memory.id,
        actor_agent_id=writer.id,
        content="The project uses SQLite.",
    )

    assert corrected.owner_agent_id == owner.id
    assert store.list_provenance(corrected.id)[0].source_agent_id == writer.id


def test_retracted_memory_is_preserved_as_history(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    old = store_memory(engine, owner_id=agent.id)

    new = engine.correct_memory(
        old.id,
        actor_agent_id=agent.id,
        content="The project uses SQLite.",
        kind=CorrectionKind.RETRACTED,
    )

    loaded_old = store.get_memory(old.id)

    assert loaded_old is not None
    assert loaded_old.status == MemoryStatus.RETRACTED
    assert loaded_old.superseded_by_id == new.id
    assert engine.list_memories(agent_id=agent.id) == [new]


def test_only_active_memory_can_be_corrected(
    engine: MemoryEngine,
) -> None:
    agent = engine.register_agent("Agent A")
    old = store_memory(engine, owner_id=agent.id)
    engine.correct_memory(
        old.id,
        actor_agent_id=agent.id,
        content="The project uses SQLite.",
    )

    with pytest.raises(InvalidStateError):
        engine.correct_memory(
            old.id,
            actor_agent_id=agent.id,
            content="The project uses DuckDB.",
        )


def test_storing_project_memory_requires_membership(
    engine: MemoryEngine,
) -> None:
    agent = engine.register_agent("Agent A")
    project = engine.create_project("Demo project")

    with pytest.raises(AccessDeniedError):
        store_memory(
            engine,
            owner_id=agent.id,
            project_id=project.id,
            scope=AccessScope.PROJECT,
        )


def test_missing_agent_and_memory_raise_not_found(
    engine: MemoryEngine,
) -> None:
    agent = engine.register_agent("Agent A")

    with pytest.raises(NotFoundError):
        engine.list_memories(agent_id=uuid4())
    with pytest.raises(NotFoundError):
        engine.get_memory(agent_id=agent.id, memory_id=uuid4())
