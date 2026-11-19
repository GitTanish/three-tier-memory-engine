"""Retrieval requests, scored hits, and assembled context."""

from pydantic import BaseModel

from memory_engine.models.memory import MemoryRecord


class RecallHit(BaseModel):
    """A memory returned by recall with its fused score and source ranks."""

    memory: MemoryRecord
    score: float
    effective_importance: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
