-- Initial schema for the local memory service.

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    project_id TEXT NOT NULL REFERENCES projects(id),
    task_id TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS project_memberships (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (project_id, agent_id)
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    owner_agent_id TEXT NOT NULL REFERENCES agents(id),
    project_id TEXT REFERENCES projects(id),
    session_id TEXT REFERENCES sessions(id),
    scope TEXT NOT NULL,
    importance REAL NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    supersedes_id TEXT REFERENCES memories(id),
    superseded_by_id TEXT REFERENCES memories(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    last_accessed_at TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_access
    ON memories(scope, project_id, owner_agent_id, status, tier);

CREATE INDEX IF NOT EXISTS idx_memories_supersedes
    ON memories(supersedes_id);

CREATE INDEX IF NOT EXISTS idx_memories_expires
    ON memories(expires_at);

CREATE TABLE IF NOT EXISTS provenance (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    source_memory_id TEXT REFERENCES memories(id),
    source_session_id TEXT REFERENCES sessions(id),
    source_agent_id TEXT REFERENCES agents(id),
    source_excerpt TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provenance_memory
    ON provenance(memory_id);
