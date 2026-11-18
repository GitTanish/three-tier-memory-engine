"""Session-scoped working memory entries."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from memory_engine.models.base import utc_now


class WorkingEntry(BaseModel):
    """A key/value entry in a session's working scratchpad.

    Working entries are mutable, private to one session, and always expire.
    They are not versioned and do not carry provenance.
    """

    session_id: UUID
    key: str = Field(min_length=1, max_length=200)
    value: Any
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
