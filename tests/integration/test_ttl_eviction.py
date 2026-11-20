"""Integration tests for TTL eviction."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from memory_engine.engine import MemoryEngine
from memory_engine.models import (
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
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


def store_memory(
    engine: MemoryEngine,
    *,
    owner_id: UUID,
    content: str = "The project uses PostgreSQL.",
    **overrides: object,
) -> MemoryRecord:
    return engine.store_memory(
        MemoryRecord(
            tier=MemoryTier.SEMANTIC,
            content=content,
            owner_agent_id=owner_id,
            **overrides,
        )
    )


def start_working_session(engine: MemoryEngine) -> tuple[UUID, UUID]:
    agent = engine.register_agent("Agent A")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, agent.id, ProjectRole.WRITER)
    session = engine.start_session(agent_id=agent.id, project_id=project.id)
    return agent.id, session.id


def test_evict_expired_marks_memories_expired(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    memory = store_memory(
        engine,
        owner_id=agent.id,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    result = engine.evict_expired()

    assert result.memories == 1
    stored = store.get_memory(memory.id)
    assert stored is not None
    assert stored.status == MemoryStatus.EXPIRED
    assert stored.content == memory.content


def test_evict_expired_removes_working_entries(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent_id, session_id = start_working_session(engine)
    engine.working.set(
        agent_id=agent_id,
        session_id=session_id,
        key="plan",
        value=["step"],
        ttl_seconds=60,
    )

    later = datetime.now(UTC) + timedelta(seconds=61)
    result = engine.evict_expired(at=later)

    assert result.working_entries == 1
    assert store.list_working_entries(session_id, at=later) == []


def test_evict_expired_leaves_unexpired_memories_alone(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    now = datetime.now(UTC)
    future = store_memory(
        engine,
        owner_id=agent.id,
        expires_at=now + timedelta(days=1),
    )
    forever = store_memory(engine, owner_id=agent.id)

    result = engine.evict_expired(at=now)

    assert result.memories == 0
    future_stored = store.get_memory(future.id)
    forever_stored = store.get_memory(forever.id)
    assert future_stored is not None
    assert forever_stored is not None
    assert future_stored.status == MemoryStatus.ACTIVE
    assert forever_stored.status == MemoryStatus.ACTIVE


def test_evict_expired_does_not_rewrite_history(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    expires_at = datetime.now(UTC) + timedelta(seconds=10)
    old = store_memory(engine, owner_id=agent.id, expires_at=expires_at)
    new = engine.correct_memory(
        old.id,
        actor_agent_id=agent.id,
        content="The project uses SQLite.",
    )

    result = engine.evict_expired(at=expires_at)

    assert result.memories == 1
    old_stored = store.get_memory(old.id)
    new_stored = store.get_memory(new.id)
    assert old_stored is not None
    assert new_stored is not None
    assert old_stored.status == MemoryStatus.SUPERSEDED
    assert new_stored.status == MemoryStatus.EXPIRED


def test_evict_expired_is_idempotent(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    store_memory(
        engine,
        owner_id=agent.id,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    assert engine.evict_expired().memories == 1
    assert engine.evict_expired().memories == 0


def test_evict_expired_uses_the_supplied_clock(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    expires_at = datetime(2026, 12, 1, tzinfo=UTC)
    memory = store_memory(engine, owner_id=agent.id, expires_at=expires_at)

    before = engine.evict_expired(at=expires_at - timedelta(seconds=1))
    at_expiry = engine.evict_expired(at=expires_at)

    assert before.memories == 0
    assert at_expiry.memories == 1
    stored = store.get_memory(memory.id)
    assert stored is not None
    assert stored.status == MemoryStatus.EXPIRED


def test_expired_memories_are_hidden_before_eviction(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    memory = store_memory(
        engine,
        owner_id=agent.id,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    assert engine.recall(agent_id=agent.id, query="postgresql") == []
    assert engine.list_memories(agent_id=agent.id) == []
    stored = store.get_memory(memory.id)
    assert stored is not None
    assert stored.status == MemoryStatus.ACTIVE
