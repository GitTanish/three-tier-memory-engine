"""Memory records, tiers, and lifecycle state."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from memory_engine.models.base import utc_now
from memory_engine.models.sharing import AccessScope


class MemoryTier(StrEnum):
    """How long a memory is expected to remain useful."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryStatus(StrEnum):
    """The lifecycle state of a memory version."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    STALE = "stale"
    EXPIRED = "expired"
    DELETED = "deleted"


class CorrectionKind(StrEnum):
    """Why an existing memory is being replaced."""

    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class MemoryRecord(BaseModel):
    """A versioned unit of memory.

    Corrections create a new record with ``supersedes_id`` pointing at the
    previous version. The old record is marked superseded or retracted and is
    retained as history.
    """

    id: UUID = Field(default_factory=uuid4)
    tier: MemoryTier
    content: str = Field(min_length=1)
    summary: str | None = None
    owner_agent_id: UUID
    project_id: UUID | None = None
    session_id: UUID | None = None
    scope: AccessScope = AccessScope.PRIVATE
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = Field(default=1, ge=1)
    supersedes_id: UUID | None = None
    superseded_by_id: UUID | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    last_accessed_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _project_scope_requires_project(self) -> "MemoryRecord":
        if self.scope == AccessScope.PROJECT and self.project_id is None:
            raise ValueError("project scope requires project_id")
        return self

    def is_expired(self, at: datetime | None = None) -> bool:
        """Return whether the memory has passed its expiration time."""
        if self.expires_at is None:
            return False
        return self.expires_at <= (at or utc_now())
