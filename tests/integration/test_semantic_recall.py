"""Integration tests for semantic recall."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from memory_engine.embeddings import HashingEmbedder
from memory_engine.engine import MemoryEngine
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


def store_memory(
    engine: MemoryEngine,
    *,
    owner_id: UUID,
    content: str,
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


def setup_project(engine: MemoryEngine) -> tuple[Agent, Project]:
    agent = engine.register_agent("Agent A")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, agent.id, ProjectRole.WRITER)
    return agent, project


def test_recall_returns_the_most_relevant_memory_first(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    store_memory(engine, owner_id=agent.id, content="The team meets on Fridays.")
    relevant = store_memory(
        engine,
        owner_id=agent.id,
        content="The project uses PostgreSQL for storage.",
    )
    store_memory(engine, owner_id=agent.id, content="The office has a coffee machine.")

    hits = engine.recall(agent_id=agent.id, query="postgresql storage")

    assert hits
    assert hits[0].memory.id == relevant.id
    assert hits[0].score > 0


def test_recall_returns_nothing_for_an_unmatched_query(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    store_memory(engine, owner_id=agent.id, content="The project uses PostgreSQL.")

    assert engine.recall(agent_id=agent.id, query="kubernetes") == []


def test_recall_filters_other_agents_private_memories(engine: MemoryEngine) -> None:
    agent_a = engine.register_agent("Agent A")
    agent_b = engine.register_agent("Agent B")
    project = engine.create_project("Demo project")
    engine.add_project_member(project.id, agent_a.id, ProjectRole.WRITER)
    engine.add_project_member(project.id, agent_b.id, ProjectRole.READER)
    private = store_memory(
        engine,
        owner_id=agent_a.id,
        content="Agent A prefers tabs.",
    )
    shared = store_memory(
        engine,
        owner_id=agent_a.id,
        content="The project uses tabs.",
        scope=AccessScope.PROJECT,
        project_id=project.id,
    )

    hits = engine.recall(agent_id=agent_b.id, query="tabs")

    assert private.id not in {hit.memory.id for hit in hits}
    assert [hit.memory.id for hit in hits] == [shared.id]


def test_recall_excludes_superseded_versions(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    old = store_memory(
        engine,
        owner_id=agent.id,
        content="The project uses PostgreSQL.",
    )
    new = engine.correct_memory(
        old.id,
        actor_agent_id=agent.id,
        content="The project uses SQLite.",
    )

    assert engine.recall(agent_id=agent.id, query="postgresql") == []
    hits = engine.recall(agent_id=agent.id, query="sqlite")
    assert [hit.memory.id for hit in hits] == [new.id]


def test_recall_filters_by_project_and_session(engine: MemoryEngine) -> None:
    agent, project = setup_project(engine)
    other_project = engine.create_project("Other project")
    engine.add_project_member(other_project.id, agent.id, ProjectRole.WRITER)
    session = engine.start_session(agent_id=agent.id, project_id=project.id)
    store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys run from the main branch.",
        scope=AccessScope.PROJECT,
        project_id=project.id,
        session_id=session.id,
    )
    store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys run from the release branch.",
        scope=AccessScope.PROJECT,
        project_id=other_project.id,
    )

    hits = engine.recall(
        agent_id=agent.id,
        query="deploys branch",
        project_id=project.id,
        session_id=session.id,
    )

    assert [hit.memory.content for hit in hits] == [
        "Deploys run from the main branch."
    ]


def test_recall_defaults_to_semantic_tier(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    semantic = store_memory(
        engine,
        owner_id=agent.id,
        content="Retries use exponential backoff.",
    )
    episodic = store_memory(
        engine,
        owner_id=agent.id,
        content="Retries were discussed in the standup.",
        tier=MemoryTier.EPISODIC,
    )

    default_hits = engine.recall(agent_id=agent.id, query="retries")
    episodic_hits = engine.recall(
        agent_id=agent.id,
        query="retries",
        tiers=[MemoryTier.EPISODIC],
    )

    assert [hit.memory.id for hit in default_hits] == [semantic.id]
    assert [hit.memory.id for hit in episodic_hits] == [episodic.id]


def test_recall_prefers_importance_then_recency(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    now = datetime.now(UTC)
    store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys need approval.",
        importance=0.1,
        created_at=now,
        updated_at=now,
    )
    important = store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys need approval.",
        importance=0.9,
        created_at=now,
        updated_at=now,
    )

    hits = engine.recall(agent_id=agent.id, query="deploys approval")

    assert hits[0].memory.id == important.id

    fresh = store_memory(
        engine,
        owner_id=agent.id,
        content="Rollbacks are manual.",
        created_at=now,
        updated_at=now,
    )
    store_memory(
        engine,
        owner_id=agent.id,
        content="Rollbacks are manual.",
        created_at=now - timedelta(days=180),
        updated_at=now - timedelta(days=180),
    )

    recency_hits = engine.recall(agent_id=agent.id, query="rollbacks manual")

    assert recency_hits[0].memory.id == fresh.id


def test_recall_respects_the_token_budget(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys run from the main branch.",
        importance=0.9,
    )
    store_memory(
        engine,
        owner_id=agent.id,
        content="Deploys are reviewed before merging.",
        importance=0.8,
    )

    hits = engine.recall(
        agent_id=agent.id,
        query="deploys",
        max_tokens=10,
    )

    assert len(hits) == 1


def test_recall_respects_the_limit(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")
    for index in range(3):
        store_memory(
            engine,
            owner_id=agent.id,
            content=f"Deploys run from branch {index}.",
        )

    assert len(engine.recall(agent_id=agent.id, query="deploys", limit=2)) == 2


def test_recall_backfills_missing_embeddings(
    engine: MemoryEngine,
    store: SqliteMemoryStore,
) -> None:
    agent = engine.register_agent("Agent A")
    memory = store.add_memory(
        MemoryRecord(
            tier=MemoryTier.SEMANTIC,
            content="The project uses PostgreSQL.",
            owner_agent_id=agent.id,
        )
    )

    hits = engine.recall(agent_id=agent.id, query="postgresql")

    assert [hit.memory.id for hit in hits] == [memory.id]
    embeddings = store.get_memory_embeddings([memory.id])
    assert embeddings[memory.id][0] == engine._embedder.id


def test_recall_rebuilds_embeddings_from_another_embedder(
    store: SqliteMemoryStore,
) -> None:
    agent = MemoryEngine(store).register_agent("Agent A")
    old_engine = MemoryEngine(store, embedder=HashingEmbedder(dimension=64))
    memory = old_engine.store_memory(
        MemoryRecord(
            tier=MemoryTier.SEMANTIC,
            content="The project uses PostgreSQL.",
            owner_agent_id=agent.id,
        )
    )

    new_engine = MemoryEngine(store, embedder=HashingEmbedder(dimension=128))
    hits = new_engine.recall(agent_id=agent.id, query="postgresql")

    assert [hit.memory.id for hit in hits] == [memory.id]
    assert store.get_memory_embeddings([memory.id])[memory.id][0] == (
        new_engine._embedder.id
    )


def test_recall_rejects_invalid_requests(engine: MemoryEngine) -> None:
    agent = engine.register_agent("Agent A")

    with pytest.raises(ValueError):
        engine.recall(agent_id=agent.id, query="   ")
    with pytest.raises(ValueError):
        engine.recall(agent_id=agent.id, query="deploys", limit=0)
    with pytest.raises(ValueError):
        engine.recall(agent_id=agent.id, query="deploys", max_tokens=0)
