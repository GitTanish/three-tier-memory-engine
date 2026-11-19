"""Integration tests for usage-based importance scoring."""

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID

import pytest

from memory_engine.engine import MemoryEngine
from memory_engine.models import MemoryRecord, MemoryTier
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


def test_recall_records_access_on_returned_hits(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    memory = store_memory(engine, owner_id=agent.id)

    engine.recall(agent_id=agent.id, query="postgresql")
    first = store.get_memory(memory.id)
    engine.recall(agent_id=agent.id, query="postgresql")
    second = store.get_memory(memory.id)

    assert first is not None
    assert first.access_count == 1
    assert first.last_accessed_at is not None
    assert second is not None
    assert second.access_count == 2


def test_only_returned_hits_count_as_access(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    memories = [
        store_memory(
            engine,
            owner_id=agent.id,
            content=f"Deploys run from branch {index}.",
        )
        for index in range(3)
    ]

    engine.recall(agent_id=agent.id, query="deploys", limit=1)

    counts = [store.get_memory(memory.id).access_count for memory in memories]
    assert sum(counts) == 1


def test_usage_can_reorder_equal_importance(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    now = datetime.now(UTC)
    first = store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys use blue green.",
        id=UUID(int=1),
        created_at=now,
        updated_at=now,
    )
    second = store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys use blue green.",
        id=UUID(int=2),
        created_at=now,
        updated_at=now,
    )

    baseline = engine.recall(agent_id=agent.id, query="deploys blue green", limit=1)
    assert baseline[0].memory.id == first.id

    for _ in range(5):
        store.record_memory_access([second.id])

    hits = engine.recall(agent_id=agent.id, query="deploys blue green", limit=2)

    assert hits[0].memory.id == second.id
    assert hits[0].effective_importance > hits[1].effective_importance


def test_correction_starts_with_fresh_usage(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    memory = store_memory(engine, owner_id=agent.id)
    engine.recall(agent_id=agent.id, query="postgresql")

    corrected = engine.correct_memory(
        memory.id,
        actor_agent_id=agent.id,
        content="The project uses SQLite.",
    )

    stored = store.get_memory(corrected.id)
    assert stored is not None
    assert stored.access_count == 0
    assert stored.last_accessed_at is None
