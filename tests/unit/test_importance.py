"""Tests for usage-based importance scoring."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from memory_engine.importance import effective_importance
from memory_engine.models import MemoryRecord, MemoryTier

NOW = datetime(2026, 11, 19, tzinfo=UTC)


def make_memory(**overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "tier": MemoryTier.SEMANTIC,
        "content": "The project uses PostgreSQL.",
        "owner_agent_id": uuid4(),
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return MemoryRecord(**values)


def test_unused_memory_keeps_its_explicit_importance() -> None:
    memory = make_memory(importance=0.7)

    assert effective_importance(memory, NOW) == 0.7


def test_recent_use_raises_importance() -> None:
    memory = make_memory(
        importance=0.5,
        access_count=5,
        last_accessed_at=NOW - timedelta(days=1),
    )

    assert effective_importance(memory, NOW) > 0.5


def test_more_use_scores_higher() -> None:
    few = make_memory(
        importance=0.5,
        access_count=1,
        last_accessed_at=NOW,
    )
    many = make_memory(
        importance=0.5,
        access_count=20,
        last_accessed_at=NOW,
    )

    assert effective_importance(many, NOW) > effective_importance(few, NOW)


def test_old_use_decays() -> None:
    recent = make_memory(
        importance=0.5,
        access_count=10,
        last_accessed_at=NOW - timedelta(days=1),
    )
    old = make_memory(
        importance=0.5,
        access_count=10,
        last_accessed_at=NOW - timedelta(days=180),
    )

    assert effective_importance(old, NOW) < effective_importance(recent, NOW)


def test_usage_boost_is_bounded() -> None:
    memory = make_memory(
        importance=0.95,
        access_count=1000,
        last_accessed_at=NOW,
    )

    assert effective_importance(memory, NOW) == 1.0
