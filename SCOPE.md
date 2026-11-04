# Scope

Status: Draft

## Purpose

Build a local, persistent memory service for multiple AI agents. The service stores what agents learn across sessions, keeps that knowledge correctable, and retrieves it under an explicit token budget.

Success is behavioral, not volumetric: an agent's tenth session on a project should require fewer corrections and less rediscovery than its first.

## Core behavior

Agent A learns that a project uses PostgreSQL.

A user corrects it: the project uses SQLite.

Later, Agent B asks which database the project uses.

The engine returns SQLite, retains the correction's provenance, and preserves PostgreSQL as historical information.

## Decisions

- **Deployment:** local service with a Python client.
- **Writes:** agents explicitly request writes; the engine does not decide on its own to remember.
- **Consolidation:** the application triggers it; the engine handles the processing.
- **Tenancy:** multi-agent from V1, with access controls.
- **Working memory:** task/session-scoped state with expiration.
- **Corrections:** explicit corrections first; automatic contradiction detection later.
- **History:** memories retain their source and history. Corrections supersede outdated knowledge rather than overwriting it.
- **Evaluation:** a coding agent working across multiple sessions is the first real-world scenario.

## Tier semantics

- **Working memory** is mutable, private to a session/task, and expires. It is scratch state, not durable knowledge.
- **Episodic memory** is an append-only record of what happened: observations, actions, decisions, and outcomes.
- **Semantic memory** is durable, versioned knowledge extracted from experience. It is the layer other agents are most likely to share.

## Architecture shape

- The service owns SQLite. Clients communicate through the API rather than reading or writing the database directly.
- SQLite is the source of truth. Retrieval indexes are derived state and must be rebuildable.
- Agent identity and project membership are checked server-side. Access filtering happens before retrieval ranking, so unauthorized memories neither appear nor influence results.
- Retrieval combines dense and sparse search, fuses the rankings, and adjusts for relevance, recency, and importance.
- Context assembly returns selected memories within a caller-supplied token budget.
- Consolidation, embeddings, and ranking are pluggable so the engine can be tested without external models.

## Correction semantics

- Memories are versioned, not mutated in place.
- A correction creates a new version, marks the old version as superseded, and records who corrected it, when, and why.
- Corrections distinguish **retracted** (the old belief was wrong) from **superseded** (the world changed).
- Normal retrieval returns current knowledge only. A separate history query returns the version chain and provenance.
- Memories derived from another memory carry a `derived_from` link. When a source is corrected, downstream memories are marked `stale` or `needs_review` rather than silently rewritten.
- Episodic records are append-only. A correction is a new event, not an edit to history.

## Access model

V1 uses scope-based access:

- **Private:** visible only to the owning agent.
- **Project:** visible to agents with membership in that project.
- **Admin:** can manage agents, projects, and grants.

Per-memory ACLs are deferred. The service enforces authorization, but a local agent with filesystem access can bypass it by opening SQLite directly. V1 treats this as cooperative isolation unless a stronger boundary is required.

## Non-goals for V1

- Autonomous memory formation.
- Automatic contradiction detection.
- Per-memory ACLs.
- Distributed or cloud synchronization.
- A user interface.
- Training or fine-tuning embedding models.

## Acceptance test

1. Register Agents A and B in the same project.
2. A writes "the project uses PostgreSQL" as a semantic memory with a source.
3. B retrieves the memory.
4. A user correction creates "the project uses SQLite," supersedes the old memory, and records the correction.
5. B's normal retrieval returns SQLite only.
6. B's history query returns PostgreSQL plus the correction chain.
7. Agent C, with no access, receives nothing.
8. If B had stored a derived copy of PostgreSQL, that copy is marked stale.

## Open questions

1. **Consolidation processing:** does the engine call an LLM to extract facts from episodic records, or does the application supply structured facts and the engine handle deduplication, provenance, versioning, and conflict flags?
2. **Correction propagation:** should downstream derived memories be marked stale automatically, or should the correction only affect the direct memory until an agent rechecks?
3. **Access-control boundary:** is cooperative isolation sufficient, or must the service protect memory from other local processes?
4. **Session end:** should working memory simply expire, or be promoted into episodic memory before the session closes?
5. **Transport and integration:** local HTTP is the default; an MCP adapter may be added later for coding agents.
6. **Embeddings:** local model, remote provider, or client-supplied vectors?

## Next step

Define the data model and API contract from this scope.
