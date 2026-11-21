# Three-Tier Memory Engine

I'm trying to give AI agents memory that persists across interactions and sessions.

Agents accumulate material as they work: observations, decisions, failures, things learned along the way. Some of it matters later. Most of it disappears when a session ends, and what remains gets harder to retrieve as context grows. This project is an attempt at a reusable memory layer for multiple agents, one that treats stored information as something with a lifecycle.

## Tiers

Memory is divided by how long it stays useful.

**Working memory** holds what the current task needs, including intermediate state.

**Episodic memory** records what happened: interactions, events, decisions, outcomes.

**Semantic memory** is the durable layer, holding facts, concepts, and relationships extracted from experience.

The engine manages movement between these tiers and decides when information should expire.

## Direction

Memory should be shareable across agents, with explicit ownership and boundaries around access. Storage is local and persistent. Retrieval mixes dense and sparse search, ranked by relevance, recency, and importance.

Consolidation, the shift from episodic record to semantic knowledge, begins as an application-triggered operation. The engine does not decide on its own that a given moment deserves to be remembered.

Updates, conflicts, provenance, and expiration belong in the model from the start. Context is assembled under a token budget.

## What would make this useful

I want an agent's tenth session on a project to go better because of the first nine. It should need fewer corrections and spend less time rediscovering things we already worked out. That's what I want to measure, not how much it can store.

The risk is that it gets better at repeating its mistakes. An incorrect assumption saved in one session can become the starting point for the next five. If another agent copies it, it can start to look independently confirmed.

So I want to test corrections across sessions, too. Tell the agent it was wrong, start a new session, and see whether it still makes the same mistake.

## Shape

Agents use a common interface to store, retrieve, update, delete, share, and consolidate memories. A central manager coordinates storage, retrieval, access control, and lifecycle behind that interface.

## Stack

Python, Pydantic, SQLite, BM25, Reciprocal Rank Fusion, pytest. These choices are provisional.

## Try it

There is no server or CLI yet. The engine is a Python library.

```python
from memory_engine import MemoryEngine
from memory_engine.models import MemoryRecord, MemoryTier
from memory_engine.storage import SqliteMemoryStore

store = SqliteMemoryStore("memory.db")
engine = MemoryEngine(store)

agent = engine.register_agent("planner")
engine.store_memory(
    MemoryRecord(
        tier=MemoryTier.SEMANTIC,
        content="The project uses SQLite as its database.",
        owner_agent_id=agent.id,
    )
)

hits = engine.recall(agent_id=agent.id, query="which database?")
print(hits[0].memory.content)
```

The default embedder matches words, not meaning. Include likely query terms in the memory, or plug in a real `Embedder` later.

## Questions

What should be remembered, how it should be represented, how agents share it safely, and how the system copes with knowledge that changes. Retrieval quality, memory consistency, and context efficiency ought to be measured rather than shown.

## Status

Work in progress. All six planned V1 components are implemented: working memory, episodic compression, semantic recall, importance scoring, TTL eviction, and session close. Next is an integration surface (local HTTP or MCP) and an evaluation harness.
