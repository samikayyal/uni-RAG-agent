"""Deterministic LangChain embedding implementation for offline tests."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

TEST_EMBEDDING_DIMENSIONS = {
    # Keep enough dimensions for stable Chroma HNSW filtering while still
    # making tests substantially lighter than real embedding models.
    "BAAI/bge-m3": 384,
    "jinaai/jina-embeddings-v3": 384,
    "jinaai/jina-embeddings-v5-text-small": 384,
    "google/embeddinggemma-300m": 768,
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class DeterministicTestEmbeddings(Embeddings):
    """Small, normalized token vectors used only at the model-loader boundary."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Test embedding dimension must be greater than zero.")
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            vector[index] += 1.0

        if not tokens:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            vector[index] = 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Preserve the embedding interface's nonempty-vector contract if
            # a future test-vector strategy ever produces a zero norm.
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            vector[index] = 1.0
            norm = 1.0
        return [value / norm for value in vector]


def embeddings_for_model(model_name: str) -> DeterministicTestEmbeddings:
    """Build the configured test dimension for a legitimate registry model."""
    try:
        dimension = TEST_EMBEDDING_DIMENSIONS[model_name]
    except KeyError as exc:
        raise ValueError(f"No test embedding dimension for {model_name!r}") from exc
    return DeterministicTestEmbeddings(dimension)
