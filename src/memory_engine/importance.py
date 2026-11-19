"""Importance scoring from explicit weight and observed use.

Callers set a base importance when they write a memory. Recall reinforces the
memories it returns: every returned hit gets one access. The effective score
adds a bounded boost for frequent, recent use, so usage can break ties without
overriding explicit intent.
"""

import math
from datetime import datetime

from memory_engine.models import MemoryRecord
from memory_engine.models.base import utc_now

USAGE_WEIGHT = 0.2
USAGE_SCALE = 5.0
ACCESS_RECENCY_HALF_LIFE_DAYS = 30.0


def effective_importance(
    memory: MemoryRecord,
    at: datetime | None = None,
) -> float:
    """Return explicit importance plus a bounded boost for recent use."""
    if memory.access_count <= 0 or memory.last_accessed_at is None:
        return memory.importance

    frequency = 1.0 - math.exp(-memory.access_count / USAGE_SCALE)
    age_days = max(
        0.0,
        ((at or utc_now()) - memory.last_accessed_at).total_seconds() / 86400.0,
    )
    recency = math.pow(0.5, age_days / ACCESS_RECENCY_HALF_LIFE_DAYS)
    return min(1.0, memory.importance + USAGE_WEIGHT * frequency * recency)
