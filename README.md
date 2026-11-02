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

## Shape

Agents use a common interface to store, retrieve, update, delete, share, and consolidate memories. A central manager coordinates storage, retrieval, access control, and lifecycle behind that interface.

## Stack

Python, Pydantic, SQLite, FAISS, BM25, Reciprocal Rank Fusion, pytest. These choices are provisional.

## Questions

What should be remembered, how it should be represented, how agents share it safely, and how the system copes with knowledge that changes. Retrieval quality, memory consistency, and context efficiency ought to be measured rather than shown.

## Status

Work in progress. The memory model, storage contracts, and lifecycle semantics come first; retrieval and multi-agent integration follow.
