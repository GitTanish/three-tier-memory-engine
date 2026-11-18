"""Deterministic episodic compression.

The application decides when a session is worth compressing. This module
turns the session's working state and episodic records into one compact
episode. It does not call a model.
"""

import json

from memory_engine.models import (
    AccessScope,
    Agent,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    Project,
    Provenance,
    ProvenanceRelation,
    Session,
    WorkingEntry,
)
from memory_engine.models.base import utc_now

MAX_ITEMS = 20
MAX_ITEM_CHARS = 200


def _truncate(text: str, limit: int = MAX_ITEM_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_value(value: object) -> str:
    try:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        text = repr(value)
    return _truncate(text)


def build_session_episode(
    *,
    session: Session,
    agent: Agent,
    project: Project,
    working_entries: list[WorkingEntry],
    episodic_memories: list[MemoryRecord],
    importance: float | None = None,
) -> tuple[MemoryRecord, list[Provenance]]:
    """Build one deterministic episode from a session's state and events."""
    if not working_entries and not episodic_memories:
        raise ValueError("session has nothing to compress")

    selected_working = sorted(
        working_entries,
        key=lambda entry: (-entry.importance, entry.key),
    )[:MAX_ITEMS]
    selected_working.sort(key=lambda entry: entry.key)
    omitted_working = len(working_entries) - len(selected_working)

    selected_events = sorted(
        episodic_memories,
        key=lambda memory: (-memory.importance, memory.created_at, str(memory.id)),
    )[:MAX_ITEMS]
    selected_events.sort(key=lambda memory: (memory.created_at, str(memory.id)))
    omitted_events = len(episodic_memories) - len(selected_events)

    task_label = session.task_id or str(session.id)
    started_at = session.started_at.isoformat()
    ended_at = session.ended_at.isoformat() if session.ended_at else "open"

    lines = [
        f"Session episode: {task_label}",
        f"Agent: {agent.name}",
        f"Project: {project.name}",
        f"Started: {started_at}",
        f"Ended: {ended_at}",
    ]

    if selected_working:
        lines.extend(["", "Working state:"])
        for entry in selected_working:
            lines.append(f"- {entry.key}: {_format_value(entry.value)}")
        if omitted_working:
            lines.append(f"- ... {omitted_working} more working entries")

    if selected_events:
        lines.extend(["", "Events:"])
        for memory in selected_events:
            lines.append(
                f"- {_truncate(memory.content)} "
                f"[importance={memory.importance:.2f}]"
            )
        if omitted_events:
            lines.append(f"- ... {omitted_events} more events")

    content = "\n".join(lines)
    summary = (
        f"Session {task_label}: {len(working_entries)} working entries, "
        f"{len(episodic_memories)} events."
    )

    source_importances = [entry.importance for entry in working_entries] + [
        memory.importance for memory in episodic_memories
    ]
    episode_importance = (
        importance
        if importance is not None
        else max(source_importances, default=0.5)
    )

    now = utc_now()
    tags = ["episode", "session"]
    if session.task_id:
        tags.append(f"task:{session.task_id}")

    episode = MemoryRecord(
        tier=MemoryTier.EPISODIC,
        content=content,
        summary=summary,
        owner_agent_id=agent.id,
        project_id=project.id,
        session_id=session.id,
        scope=AccessScope.PROJECT,
        importance=episode_importance,
        confidence=1.0,
        status=MemoryStatus.ACTIVE,
        version=1,
        created_at=now,
        updated_at=now,
        tags=tags,
        metadata={
            "kind": "session_episode",
            "source_session_id": str(session.id),
            "task_id": session.task_id,
            "working_keys": [entry.key for entry in working_entries],
            "episodic_memory_ids": [
                str(memory.id) for memory in episodic_memories
            ],
            "working_omitted": omitted_working,
            "events_omitted": omitted_events,
            "started_at": started_at,
            "ended_at": (
                session.ended_at.isoformat() if session.ended_at else None
            ),
        },
    )

    provenance = [
        Provenance(
            memory_id=episode.id,
            relation=ProvenanceRelation.COMPRESSED_FROM,
            source_session_id=session.id,
            source_agent_id=agent.id,
            source_excerpt=summary,
        )
    ]
    provenance.extend(
        Provenance(
            memory_id=episode.id,
            relation=ProvenanceRelation.COMPRESSED_FROM,
            source_memory_id=memory.id,
            source_agent_id=agent.id,
            source_excerpt=_truncate(memory.content),
        )
        for memory in episodic_memories
    )
    return episode, provenance
