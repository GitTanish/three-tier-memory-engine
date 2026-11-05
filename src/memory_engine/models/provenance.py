"""Memory origin and derivation chains."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from memory_engine.models.base import utc_now


class ProvenanceRelation(StrEnum):
    """How a memory relates to its source."""

    OBSERVED_IN = "observed_in"
    DERIVED_FROM = "derived_from"
    CORRECTED_BY = "corrected_by"
    COPIED_FROM = "copied_from"
    CONSOLIDATED_FROM = "consolidated_from"


class Provenance(BaseModel):
    """An edge from a memory back to what produced or changed it."""

    id: UUID = Field(default_factory=uuid4)
    memory_id: UUID
    relation: ProvenanceRelation
    source_memory_id: UUID | None = None
    source_session_id: UUID | None = None
    source_agent_id: UUID | None = None
    source_excerpt: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _has_a_source(self) -> "Provenance":
        if not any(
            (self.source_memory_id, self.source_session_id, self.source_agent_id)
        ):
            raise ValueError("provenance requires at least one source")
        return self
