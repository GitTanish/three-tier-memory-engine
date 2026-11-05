"""Visibility scopes and project membership."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from memory_engine.models.base import utc_now


class AccessScope(StrEnum):
    """Who can see a memory."""

    PRIVATE = "private"
    PROJECT = "project"


class ProjectRole(StrEnum):
    """What an agent may do inside a project."""

    READER = "reader"
    WRITER = "writer"
    ADMIN = "admin"


class ProjectMembership(BaseModel):
    """An agent's role within a project."""

    project_id: UUID
    agent_id: UUID
    role: ProjectRole = ProjectRole.READER
    created_at: datetime = Field(default_factory=utc_now)
