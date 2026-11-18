"""Integration tests for the working scratchpad."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from memory_engine.engine import MemoryEngine
from memory_engine.errors import AccessDeniedError, NotFoundError
from memory_engine.models import Agent, Project, ProjectRole, Session, WorkingEntry
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


def setup_session(
    engine: MemoryEngine,
    *,
    agent_name: str = "Agent A",
) -> tuple[Agent, Project, Session]:
    agent = engine.register_agent(agent_name)
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, agent.id, ProjectRole.WRITER)
    session = engine.start_session(agent_id=agent.id, project_id=project.id)
    return agent, project, session


def test_set_and_get_round_trip(engine: MemoryEngine) -> None:
    agent, _project, session = setup_session(engine)

    entry = engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value={"step": 1, "files": ["engine.py"]},
    )
    loaded = engine.working.get(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
    )

    assert loaded is not None
    assert loaded.value == {"step": 1, "files": ["engine.py"]}
    assert loaded.key == "plan"
    assert loaded.importance == 0.5
    assert loaded.expires_at > loaded.created_at
    assert entry.created_at == loaded.created_at


def test_set_replaces_existing_key_and_preserves_created_at(
    engine: MemoryEngine,
) -> None:
    agent, _project, session = setup_session(engine)

    first = engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="first",
    )
    second = engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="second",
    )
    entries = engine.working.list(agent_id=agent.id, session_id=session.id)

    assert len(entries) == 1
    assert entries[0].value == "second"
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_entries_are_session_scoped(engine: MemoryEngine) -> None:
    agent, project, session_one = setup_session(engine)
    session_two = engine.start_session(agent_id=agent.id, project_id=project.id)

    engine.working.set(
        agent_id=agent.id,
        session_id=session_one.id,
        key="plan",
        value="session one only",
    )

    assert (
        engine.working.get(
            agent_id=agent.id,
            session_id=session_two.id,
            key="plan",
        )
        is None
    )
    assert engine.working.list(agent_id=agent.id, session_id=session_two.id) == []


def test_other_agent_cannot_access_session(engine: MemoryEngine) -> None:
    agent, _project, session = setup_session(engine)
    other = engine.register_agent("Agent B")
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="private",
    )

    with pytest.raises(AccessDeniedError):
        engine.working.get(
            agent_id=other.id,
            session_id=session.id,
            key="plan",
        )


def test_expired_entries_are_hidden_and_evicted(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent, _project, session = setup_session(engine)
    store.upsert_working_entry(
        WorkingEntry(
            session_id=session.id,
            key="old",
            value="stale",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )

    assert (
        engine.working.get(
            agent_id=agent.id,
            session_id=session.id,
            key="old",
        )
        is None
    )
    assert engine.working.list(agent_id=agent.id, session_id=session.id) == []
    assert engine.working.evict_expired() == 1
    assert store.get_working_entry(session.id, "old") is None


def test_invalid_ttl_and_value_are_rejected(engine: MemoryEngine) -> None:
    agent, _project, session = setup_session(engine)

    with pytest.raises(ValueError):
        engine.working.set(
            agent_id=agent.id,
            session_id=session.id,
            key="plan",
            value="x",
            ttl_seconds=0,
        )
    with pytest.raises(ValueError):
        engine.working.set(
            agent_id=agent.id,
            session_id=session.id,
            key="plan",
            value=object(),
        )


def test_delete_and_clear(engine: MemoryEngine) -> None:
    agent, _project, session = setup_session(engine)
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="a",
        value=1,
    )
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="b",
        value=2,
    )

    assert engine.working.delete(agent_id=agent.id, session_id=session.id, key="a")
    remaining = engine.working.list(agent_id=agent.id, session_id=session.id)
    assert [entry.key for entry in remaining] == ["b"]
    assert engine.working.clear(agent_id=agent.id, session_id=session.id) == 1
    assert engine.working.list(agent_id=agent.id, session_id=session.id) == []


def test_scratchpad_persists_across_store_restarts(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"

    with SqliteMemoryStore(database) as store:
        engine = MemoryEngine(store)
        agent = engine.register_agent("Agent A")
        project = engine.create_project("Demo project")
        engine.add_project_member(project.id, agent.id, ProjectRole.WRITER)
        session = engine.start_session(agent_id=agent.id, project_id=project.id)
        engine.working.set(
            agent_id=agent.id,
            session_id=session.id,
            key="plan",
            value={"step": 2},
            ttl_seconds=3600,
        )

    with SqliteMemoryStore(database) as store:
        engine = MemoryEngine(store)
        entry = engine.working.get(
            agent_id=agent.id,
            session_id=session.id,
            key="plan",
        )

    assert entry is not None
    assert entry.value == {"step": 2}


def test_start_session_requires_project_membership(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    project = engine.create_project("Demo project")

    with pytest.raises(AccessDeniedError):
        engine.start_session(agent_id=agent.id, project_id=project.id)


def test_missing_session_raises_not_found(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")

    with pytest.raises(NotFoundError):
        engine.working.list(agent_id=agent.id, session_id=uuid4())
