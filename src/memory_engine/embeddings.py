"""Deterministic text embeddings for local recall.

The embedder is pluggable. The hashing implementation gives the retrieval
pipeline a dense signal without a model download or a remote dependency.
"""

import hashlib
import math
import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class Embedder(Protocol):
    """Turns text into a fixed-length vector."""

    id: str
    dimension: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Hashing bag-of-words embedder with a stable, process-independent hash.

    shortcut: hashing matches tokens, not meaning; swap in a local model when
    recall quality matters more than zero setup.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self.dimension = dimension
        self.id = f"hashing-v1:{dimension}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
