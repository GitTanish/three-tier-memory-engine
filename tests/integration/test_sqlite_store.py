"""Integration tests for the SQLite store."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

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
from memory_engine.storage import SqliteMemoryStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store() -> Iterator[SqliteMemoryStore]:
    database = SqliteMemoryStore()
    yield database
    database.close()


def add_agent(store: SqliteMemoryStore, name: str) -> Agent:
    return store.add_agent(Agent(name=name))


def add_project(store: SqliteMemoryStore, name: str = "Demo project") -> Project:
    return store.add_project(Project(name=name))


def add_memory(
    store: SqliteMemoryStore,
    owner: Agent,
    *,
    content: str = "The project uses PostgreSQL.",
    tier: MemoryTier = MemoryTier.SEMANTIC,
    **overrides: object,
) -> MemoryRecord:
    return store.add_memory(
        MemoryRecord(
            tier=tier,
            content=content,
            owner_agent_id=owner.id,
            **overrides,
        )
    )


def test_memory_round_trip(store: SqliteMemoryStore) -> None:
    agent = add_agent(store, "Agent A")
    memory = add_memory(
        store,
        agent,
        tags=["database", "project"],
        metadata={"source": "session-1"},
    )

    loaded = store.get_memory(memory.id)

    assert loaded == memory


def test_agent_project_session_and_membership_round_trip(
    store: SqliteMemoryStore,
) -> None:
    agent = add_agent(store, "Agent A")
    project = add_project(store)
    session = store.add_session(
        Session(agent_id=agent.id, project_id=project.id, task_id="task-1")
    )
    membership = store.add_membership(
        ProjectMembership(
            project_id=project.id,
            agent_id=agent.id,
            role=ProjectRole.WRITER,
        )
    )

    assert store.get_agent(agent.id) == agent
    assert store.get_project(project.id) == project
    assert store.get_session(session.id) == session
    assert store.get_membership(project.id, agent.id) == membership


def test_private_memory_is_visible_only_to_owner(store: SqliteMemoryStore) -> None:
    owner = add_agent(store, "Agent A")
    other = add_agent(store, "Agent B")
    memory = add_memory(store, owner)

    assert store.list_visible_memories(agent_id=owner.id) == [memory]
    assert store.list_visible_memories(agent_id=other.id) == []


def test_project_memory_is_visible_to_project_members(
    store: SqliteMemoryStore,
) -> None:
    owner = add_agent(store, "Agent A")
    member = add_agent(store, "Agent B")
    outsider = add_agent(store, "Agent C")
    project = add_project(store)
    store.add_membership(
        ProjectMembership(
            project_id=project.id,
            agent_id=member.id,
            role=ProjectRole.READER,
        )
    )
    memory = add_memory(
        store,
        owner,
        project_id=project.id,
        scope=AccessScope.PROJECT,
    )

    assert store.list_visible_memories(agent_id=owner.id) == [memory]
    assert store.list_visible_memories(agent_id=member.id) == [memory]
    assert store.list_visible_memories(agent_id=outsider.id) == []


def test_expired_memory_is_hidden(store: SqliteMemoryStore) -> None:
    agent = add_agent(store, "Agent A")
    add_memory(
        store,
        agent,
        tier=MemoryTier.WORKING,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    assert store.list_visible_memories(agent_id=agent.id) == []


def test_correction_history_and_current_visibility(
    store: SqliteMemoryStore,
) -> None:
    agent = add_agent(store, "Agent A")
    old = add_memory(store, agent, content="The project uses PostgreSQL.")
    new = add_memory(
        store,
        agent,
        content="The project uses SQLite.",
        version=2,
        supersedes_id=old.id,
    )
    store.set_memory_status(
        old.id,
        MemoryStatus.SUPERSEDED,
        superseded_by_id=new.id,
    )

    visible = store.list_visible_memories(agent_id=agent.id)
    history = store.get_history(new.id)

    assert [memory.content for memory in visible] == ["The project uses SQLite."]
    assert [memory.content for memory in history] == [
        "The project uses PostgreSQL.",
        "The project uses SQLite.",
    ]
    assert history[0].status == MemoryStatus.SUPERSEDED
    assert history[1].status == MemoryStatus.ACTIVE


def test_tier_filter(store: SqliteMemoryStore) -> None:
    agent = add_agent(store, "Agent A")
    semantic = add_memory(store, agent, content="A durable fact.")
    working = add_memory(
        store,
        agent,
        content="Current task state.",
        tier=MemoryTier.WORKING,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    visible = store.list_visible_memories(
        agent_id=agent.id,
        tiers=[MemoryTier.WORKING],
    )

    assert visible == [working]
    assert semantic not in visible


def test_provenance_round_trip(store: SqliteMemoryStore) -> None:
    agent = add_agent(store, "Agent A")
    old = add_memory(store, agent, content="The project uses PostgreSQL.")
    new = add_memory(
        store,
        agent,
        content="The project uses SQLite.",
        version=2,
        supersedes_id=old.id,
    )
    provenance = Provenance(
        memory_id=new.id,
        relation=ProvenanceRelation.CORRECTED_BY,
        source_memory_id=old.id,
        source_agent_id=agent.id,
    )

    store.add_provenance(provenance)

    assert store.list_provenance(new.id) == [provenance]


def test_list_visible_memories_rejects_invalid_limit(
    store: SqliteMemoryStore,
) -> None:
    agent = add_agent(store, "Agent A")

    with pytest.raises(ValueError):
        store.list_visible_memories(agent_id=agent.id, limit=0)


def test_get_missing_memory_returns_none(store: SqliteMemoryStore) -> None:
    assert store.get_memory(UUID(int=0)) is None


def test_file_backed_store_persists(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"

    with SqliteMemoryStore(database) as store:
        agent = add_agent(store, "Agent A")
        memory = add_memory(store, agent)

    with SqliteMemoryStore(database) as store:
        assert store.get_agent(agent.id) == agent
        assert store.get_memory(memory.id) == memory
