"""Integration tests for episodic compression."""

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memory_engine.compression import build_session_episode
from memory_engine.engine import MemoryEngine
from memory_engine.errors import AccessDeniedError, InvalidStateError, NotFoundError
from memory_engine.models import (
    AccessScope,
    Agent,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    Project,
    ProjectRole,
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


@pytest.fixture
def engine(store: SqliteMemoryStore) -> MemoryEngine:
    return MemoryEngine(store)


def setup_workspace(
    engine: MemoryEngine,
    *,
    task_id: str = "T-1",
) -> tuple[Agent, Project, Session]:
    agent = engine.register_agent("Agent A")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, agent.id, ProjectRole.WRITER)
    session = engine.start_session(
        agent_id=agent.id,
        project_id=project.id,
        task_id=task_id,
    )
    return agent, project, session


def add_episodic(
    engine: MemoryEngine,
    *,
    agent: Agent,
    project: Project,
    session: Session,
    content: str,
    importance: float = 0.5,
    created_at: datetime | None = None,
) -> MemoryRecord:
    now = created_at or datetime.now(UTC)
    return engine.store_memory(
        MemoryRecord(
            tier=MemoryTier.EPISODIC,
            content=content,
            owner_agent_id=agent.id,
            project_id=project.id,
            session_id=session.id,
            scope=AccessScope.PROJECT,
            importance=importance,
            created_at=now,
            updated_at=now,
        )
    )


def test_compress_session_builds_deterministic_episode(
    engine: MemoryEngine,
) -> None:
    agent, project, session = setup_workspace(engine)
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="z-plan",
        value={"step": 2},
        importance=0.9,
    )
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="a-note",
        value="start",
        importance=0.3,
    )
    event = add_episodic(
        engine,
        agent=agent,
        project=project,
        session=session,
        content="Ran the test suite.",
        importance=0.4,
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )

    episode = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )

    assert episode.tier == MemoryTier.EPISODIC
    assert episode.owner_agent_id == agent.id
    assert episode.project_id == project.id
    assert episode.session_id == session.id
    assert episode.scope == AccessScope.PROJECT
    assert episode.status == MemoryStatus.ACTIVE
    assert episode.importance == 0.9
    assert episode.metadata["kind"] == "session_episode"
    assert episode.metadata["source_session_id"] == str(session.id)
    assert episode.metadata["task_id"] == "T-1"
    assert episode.metadata["working_keys"] == ["a-note", "z-plan"]
    assert episode.metadata["episodic_memory_ids"] == [str(event.id)]
    assert episode.metadata["working_omitted"] == 0
    assert episode.metadata["events_omitted"] == 0
    assert "Working state:" in episode.content
    assert episode.content.index("- a-note:") < episode.content.index("- z-plan:")
    assert "Ran the test suite." in episode.content
    assert episode.summary == "Session T-1: 2 working entries, 1 events."

    working_entries = engine.working.list(
        agent_id=agent.id,
        session_id=session.id,
    )
    rebuilt, _ = build_session_episode(
        session=session,
        agent=agent,
        project=project,
        working_entries=working_entries,
        episodic_memories=[event],
    )
    assert rebuilt.content == episode.content


def test_compress_session_records_provenance(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent, project, session = setup_workspace(engine)
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="ship it",
    )
    event = add_episodic(
        engine,
        agent=agent,
        project=project,
        session=session,
        content="Deployed the service.",
    )

    episode = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )
    edges = store.list_provenance(episode.id)

    assert len(edges) == 2
    assert {edge.relation for edge in edges} == {ProvenanceRelation.COMPRESSED_FROM}
    assert all(edge.source_agent_id == agent.id for edge in edges)
    session_edges = [edge for edge in edges if edge.source_session_id == session.id]
    memory_edges = [edge for edge in edges if edge.source_memory_id == event.id]
    assert len(session_edges) == 1
    assert len(memory_edges) == 1


def test_compress_session_is_idempotent(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent, project, session = setup_workspace(engine)
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="ship it",
    )
    add_episodic(
        engine,
        agent=agent,
        project=project,
        session=session,
        content="Deployed the service.",
    )

    first = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )
    second = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )
    episodes = [
        memory
        for memory in engine.list_memories(
            agent_id=agent.id,
            project_id=project.id,
            session_id=session.id,
            tiers=[MemoryTier.EPISODIC],
            limit=None,
        )
        if memory.metadata.get("kind") == "session_episode"
    ]

    assert second.id == first.id
    assert [memory.id for memory in episodes] == [first.id]
    assert len(store.list_provenance(first.id)) == 2


def test_compress_empty_session_raises(engine: MemoryEngine) -> None:
    agent, _project, session = setup_workspace(engine)

    with pytest.raises(InvalidStateError, match="nothing to compress"):
        engine.compress_session(agent_id=agent.id, session_id=session.id)


def test_only_session_owner_can_compress(engine: MemoryEngine) -> None:
    owner, project, session = setup_workspace(engine)
    other = engine.register_agent("Agent B")
    engine.add_project_member(project.id, other.id, ProjectRole.WRITER)
    engine.working.set(
        agent_id=owner.id,
        session_id=session.id,
        key="plan",
        value="private",
    )

    with pytest.raises(AccessDeniedError, match="cannot compress"):
        engine.compress_session(agent_id=other.id, session_id=session.id)

    episode = engine.compress_session(agent_id=owner.id, session_id=session.id)
    assert episode.session_id == session.id


def test_compression_uses_only_memories_from_the_session(
    engine: MemoryEngine,
) -> None:
    agent, project, session_one = setup_workspace(engine)
    session_two = engine.start_session(
        agent_id=agent.id,
        project_id=project.id,
        task_id="T-2",
    )
    first = add_episodic(
        engine,
        agent=agent,
        project=project,
        session=session_one,
        content="First session event.",
    )
    add_episodic(
        engine,
        agent=agent,
        project=project,
        session=session_two,
        content="Second session event.",
    )

    episode = engine.compress_session(
        agent_id=agent.id,
        session_id=session_one.id,
    )

    assert episode.metadata["episodic_memory_ids"] == [str(first.id)]
    assert "First session event." in episode.content
    assert "Second session event." not in episode.content


def test_compression_keeps_sources_and_working_state(
    engine: MemoryEngine,
) -> None:
    agent, project, session = setup_workspace(engine)
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="ship it",
    )
    event = add_episodic(
        engine,
        agent=agent,
        project=project,
        session=session,
        content="Deployed the service.",
    )

    episode = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )

    assert episode.id != event.id
    assert (
        engine.working.get(
            agent_id=agent.id,
            session_id=session.id,
            key="plan",
        )
        is not None
    )
    source = engine.get_memory(agent_id=agent.id, memory_id=event.id)
    assert source.status == MemoryStatus.ACTIVE
    assert source.content == "Deployed the service."


def test_importance_defaults_to_max_source_and_can_be_overridden(
    engine: MemoryEngine,
) -> None:
    agent, project, session = setup_workspace(engine)
    engine.working.set(
        agent_id=agent.id,
        session_id=session.id,
        key="plan",
        value="low",
        importance=0.2,
    )
    add_episodic(
        engine,
        agent=agent,
        project=project,
        session=session,
        content="Important event.",
        importance=0.7,
    )

    default = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )

    assert default.importance == 0.7

    second_session = engine.start_session(
        agent_id=agent.id,
        project_id=project.id,
        task_id="T-2",
    )
    engine.working.set(
        agent_id=agent.id,
        session_id=second_session.id,
        key="plan",
        value="low",
        importance=0.1,
    )
    add_episodic(
        engine,
        agent=agent,
        project=project,
        session=second_session,
        content="Less important event.",
        importance=0.3,
    )

    overridden = engine.compress_session(
        agent_id=agent.id,
        session_id=second_session.id,
        importance=0.95,
    )

    assert overridden.importance == 0.95


def test_project_members_can_read_episode_but_outsiders_cannot(
    engine: MemoryEngine,
) -> None:
    owner, project, session = setup_workspace(engine)
    reader = engine.register_agent("Agent B")
    engine.add_project_member(project.id, reader.id, ProjectRole.READER)
    outsider = engine.register_agent("Agent C")
    engine.working.set(
        agent_id=owner.id,
        session_id=session.id,
        key="plan",
        value="ship it",
    )

    episode = engine.compress_session(agent_id=owner.id, session_id=session.id)
    visible = engine.list_memories(
        agent_id=reader.id,
        project_id=project.id,
        tiers=[MemoryTier.EPISODIC],
        limit=None,
    )

    assert [memory.id for memory in visible] == [episode.id]
    assert (
        engine.list_memories(
            agent_id=outsider.id,
            project_id=project.id,
            tiers=[MemoryTier.EPISODIC],
            limit=None,
        )
        == []
    )


def test_compression_caps_working_items_and_values(
    engine: MemoryEngine,
) -> None:
    agent, _project, session = setup_workspace(engine)
    for index in range(25):
        engine.working.set(
            agent_id=agent.id,
            session_id=session.id,
            key=f"key-{index:02d}",
            value="x" * 500,
            importance=0.5,
        )

    episode = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )
    lines = [
        line for line in episode.content.splitlines() if line.startswith("- key-")
    ]

    assert episode.metadata["working_omitted"] == 5
    assert "... 5 more working entries" in episode.content
    assert len(lines) == 20
    assert all(len(line) <= len("- key-00: ") + 200 for line in lines)


def test_compression_caps_episodic_events(engine: MemoryEngine) -> None:
    agent, project, session = setup_workspace(engine)
    for index in range(25):
        add_episodic(
            engine,
            agent=agent,
            project=project,
            session=session,
            content=f"event-{index:02d}",
            importance=0.5,
        )

    episode = engine.compress_session(
        agent_id=agent.id,
        session_id=session.id,
    )
    lines = [
        line for line in episode.content.splitlines() if line.startswith("- event-")
    ]

    assert episode.metadata["events_omitted"] == 5
    assert "... 5 more events" in episode.content
    assert len(lines) == 20


def test_missing_session_raises_not_found(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")

    with pytest.raises(NotFoundError):
        engine.compress_session(agent_id=agent.id, session_id=uuid4())
