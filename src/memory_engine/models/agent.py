"""Agent, project, and session identity."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from memory_engine.models.base import utc_now


class Agent(BaseModel):
    """A participant that can own and access memories."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    created_at: datetime = Field(default_factory=utc_now)
    is_active: bool = True


class Project(BaseModel):
    """A shared boundary for memories and agent membership."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    created_at: datetime = Field(default_factory=utc_now)


class Session(BaseModel):
    """A bounded interaction between one agent and one project."""

    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    project_id: UUID
    task_id: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None

    @model_validator(mode="after")
    def _ended_after_started(self) -> "Session":
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at must not be earlier than started_at")
        return self
