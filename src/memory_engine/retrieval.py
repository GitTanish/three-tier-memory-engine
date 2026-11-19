"""Hybrid retrieval: BM25 lexical ranking, dense cosine ranking, RRF fusion."""

import math
import re
from collections import Counter
from datetime import datetime
from uuid import UUID

from memory_engine.embeddings import Embedder
from memory_engine.importance import effective_importance
from memory_engine.models import MemoryRecord, RecallHit
from memory_engine.models.base import utc_now

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75
IMPORTANCE_WEIGHT = 0.25
RECENCY_HALF_LIFE_DAYS = 30.0


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _lexical_scores(
    query: str,
    memories: list[MemoryRecord],
) -> dict[UUID, float]:
    query_tokens = set(tokenize(query))
    if not query_tokens or not memories:
        return {}

    documents = {memory.id: tokenize(memory.content) for memory in memories}
    document_frequency: Counter[str] = Counter()
    for tokens in documents.values():
        document_frequency.update(set(tokens))

    total_documents = len(memories)
    average_length = (
        sum(len(tokens) for tokens in documents.values()) / total_documents
    )
    if average_length == 0:
        return {}

    scores: dict[UUID, float] = {}
    for memory in memories:
        tokens = documents[memory.id]
        counts = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            document_freq = document_frequency[term]
            idf = math.log(
                1 + (total_documents - document_freq + 0.5) / (document_freq + 0.5)
            )
            denominator = frequency + BM25_K1 * (
                1 - BM25_B + BM25_B * len(tokens) / average_length
            )
            score += idf * (frequency * (BM25_K1 + 1)) / denominator
        if score > 0:
            scores[memory.id] = score
    return scores


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dense_scores(
    query: str,
    memories: list[MemoryRecord],
    embeddings: dict[UUID, list[float]],
    embedder: Embedder,
) -> dict[UUID, float]:
    query_vector = embedder.embed(query)
    scores: dict[UUID, float] = {}
    for memory in memories:
        vector = embeddings.get(memory.id)
        if vector is None:
            continue
        score = _cosine(query_vector, vector)
        if score > 0:
            scores[memory.id] = score
    return scores


def _ranks(scores: dict[UUID, float]) -> dict[UUID, int]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    return {
        memory_id: rank
        for rank, (memory_id, _score) in enumerate(ordered, start=1)
    }


def rank_memories(
    *,
    query: str,
    memories: list[MemoryRecord],
    embeddings: dict[UUID, list[float]],
    embedder: Embedder,
    now: datetime | None = None,
) -> list[RecallHit]:
    """Rank candidate memories by lexical, dense, importance, and recency."""
    lexical_ranks = _ranks(_lexical_scores(query, memories))
    dense_ranks = _ranks(_dense_scores(query, memories, embeddings, embedder))
    if not lexical_ranks and not dense_ranks:
        return []

    current = now or utc_now()
    by_id = {memory.id: memory for memory in memories}
    hits: list[RecallHit] = []
    for memory_id in set(lexical_ranks) | set(dense_ranks):
        memory = by_id[memory_id]
        score = 0.0
        if memory_id in lexical_ranks:
            score += 1.0 / (RRF_K + lexical_ranks[memory_id])
        if memory_id in dense_ranks:
            score += 1.0 / (RRF_K + dense_ranks[memory_id])

        importance = effective_importance(memory, current)
        score *= 1.0 + IMPORTANCE_WEIGHT * importance
        age_days = max(
            0.0,
            (current - memory.created_at).total_seconds() / 86400.0,
        )
        score *= 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
        hits.append(
            RecallHit(
                memory=memory,
                score=score,
                effective_importance=importance,
                lexical_rank=lexical_ranks.get(memory_id),
                dense_rank=dense_ranks.get(memory_id),
            )
        )

    hits.sort(
        key=lambda hit: (
            -hit.score,
            -hit.memory.created_at.timestamp(),
            str(hit.memory.id),
        )
    )
    return hits


def apply_token_budget(
    hits: list[RecallHit],
    max_tokens: int | None,
) -> list[RecallHit]:
    """Keep the highest-ranked hits that fit an approximate token budget."""
    if max_tokens is None:
        return hits

    selected: list[RecallHit] = []
    used = 0
    for hit in hits:
        tokens = max(1, len(hit.memory.content) // 4)
        if used + tokens > max_tokens:
            break
        selected.append(hit)
        used += tokens
    return selected
