"""Integration tests for cross-session persistence."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from memory_engine.engine import MemoryEngine
from memory_engine.errors import AccessDeniedError, NotFoundError
from memory_engine.models import (
    AccessScope,
    Agent,
    MemoryRecord,
    MemoryTier,
    Project,
    ProjectRole,
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


def setup_project(engine: MemoryEngine) -> tuple[Agent, Project]:
    agent = engine.register_agent("Agent A")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, agent.id, ProjectRole.WRITER)
    return agent, project


def store_fact(
    engine: MemoryEngine,
    *,
    agent: Agent,
    project: Project,
    session_id: UUID,
    content: str,
) -> MemoryRecord:
    return engine.store_memory(
        MemoryRecord(
            tier=MemoryTier.SEMANTIC,
            content=content,
            owner_agent_id=agent.id,
            project_id=project.id,
            session_id=session_id,
            scope=AccessScope.PROJECT,
        )
    )


def test_end_session_stamps_time_and_clears_working_state(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent, project = setup_project(engine)
    session = engine.start_session(agent_id=agent.id, project_id=project.id)
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="migrate",
    )
    ended_at = datetime.now(UTC)

    closed = engine.end_session(
        agent_id=agent.id,
        session_id=session.id,
        at=ended_at,
    )

    assert closed.ended_at == ended_at
    assert engine.working.list(agent_id=agent.id, session_id=session.id) == []
    stored = store.get_session(session.id)
    assert stored is not None
    assert stored.ended_at == ended_at


def test_end_session_is_idempotent(engine: MemoryEngine) -> None:
    agent, project = setup_project(engine)
    session = engine.start_session(agent_id=agent.id, project_id=project.id)
    first = engine.end_session(
        agent_id=agent.id,
        session_id=session.id,
        at=datetime.now(UTC),
    )

    second = engine.end_session(
        agent_id=agent.id,
        session_id=session.id,
        at=datetime.now(UTC) + timedelta(minutes=5),
    )

    assert second.ended_at == first.ended_at


def test_end_session_requires_the_owning_agent(engine: MemoryEngine) -> None:
    agent, project = setup_project(engine)
    other = engine.register_agent("Agent B")
    engine.add_project_member(project.id, other.id, ProjectRole.READER)
    session = engine.start_session(agent_id=agent.id, project_id=project.id)

    with pytest.raises(AccessDeniedError):
        engine.end_session(agent_id=other.id, session_id=session.id)


def test_end_session_rejects_unknown_sessions(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")

    with pytest.raises(NotFoundError):
        engine.end_session(agent_id=agent.id, session_id=UUID(int=0))


def test_end_session_rejects_a_time_before_the_start(
    engine: MemoryEngine,
) -> None:
    agent, project = setup_project(engine)
    session = engine.start_session(agent_id=agent.id, project_id=project.id)

    with pytest.raises(ValueError):
        engine.end_session(
            agent_id=agent.id,
            session_id=session.id,
            at=session.started_at - timedelta(seconds=1),
        )


def test_knowledge_survives_session_end(engine: MemoryEngine) -> None:
    agent, project = setup_project(engine)
    first = engine.start_session(agent_id=agent.id, project_id=project.id)
    fact = store_fact(
        engine,
        agent=agent,
        project=project,
        session_id=first.id,
        content="The project uses PostgreSQL.",
    )
    engine.working.set(
        agent_id=agent.id,
        session_id=first.id,
        key="decision",
        value="Use PostgreSQL",
    )
    episode = engine.compress_session(agent_id=agent.id, session_id=first.id)
    engine.end_session(agent_id=agent.id, session_id=first.id)

    second = engine.start_session(agent_id=agent.id, project_id=project.id)
    semantic_hits = engine.recall(
        agent_id=agent.id,
        query="postgresql",
        project_id=project.id,
    )
    episodic_hits = engine.recall(
        agent_id=agent.id,
        query="postgresql",
        project_id=project.id,
        tiers=[MemoryTier.EPISODIC],
    )

    assert second.id != first.id
    assert [hit.memory.id for hit in semantic_hits] == [fact.id]
    assert [hit.memory.id for hit in episodic_hits] == [episode.id]


def test_corrections_stick_across_sessions(engine: MemoryEngine) -> None:
    agent, project = setup_project(engine)
    first = engine.start_session(agent_id=agent.id, project_id=project.id)
    old = store_fact(
        engine,
        agent=agent,
        project=project,
        session_id=first.id,
        content="The project uses PostgreSQL.",
    )
    engine.end_session(agent_id=agent.id, session_id=first.id)

    second = engine.start_session(agent_id=agent.id, project_id=project.id)
    before = engine.recall(
        agent_id=agent.id,
        query="postgresql",
        project_id=project.id,
    )
    corrected = engine.correct_memory(
        old.id,
        actor_agent_id=agent.id,
        content="The project uses SQLite.",
    )
    engine.end_session(agent_id=agent.id, session_id=second.id)

    engine.start_session(agent_id=agent.id, project_id=project.id)
    assert engine.recall(
        agent_id=agent.id,
        query="postgresql",
        project_id=project.id,
    ) == []
    current = engine.recall(
        agent_id=agent.id,
        query="sqlite",
        project_id=project.id,
    )
    history = engine.get_history(agent_id=agent.id, memory_id=corrected.id)

    assert [hit.memory.id for hit in before] == [old.id]
    assert [hit.memory.id for hit in current] == [corrected.id]
    assert [memory.content for memory in history] == [
        "The project uses PostgreSQL.",
        "The project uses SQLite.",
    ]
