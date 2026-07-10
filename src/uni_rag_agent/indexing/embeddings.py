"""Embedding adapters and model building.

The default path is a deterministic, offline fake adapter used by tests and the
fake provider. Real Hugging Face local models are loaded lazily through the
optional ``embeddings`` extra so the default install stays lightweight.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from uni_rag_agent.config import Config

from .models import VectorIndexError
from .profiles import EmbeddingProfile, resolve_embedding_profile

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class FakeDeterministicEmbeddings(Embeddings):
    """Deterministic, offline embeddings backed only by hashing.

    Each token is hashed into a bucket of a fixed-dimension vector and the
    vector is L2-normalized. This keeps the adapter dependency-free and
    deterministic while remaining semantically meaningful enough for tests:
    identical text yields identical vectors (cosine distance 0), and texts that
    share tokens score closer than disjoint texts.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise VectorIndexError("Embedding dimension must be greater than zero.")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall(text.casefold())
        for token in tokens:
            index = self._bucket(token.encode("utf-8"))
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Defensive fallback for empty/no-token text: a single deterministic
            # unit component derived from the whole string.
            vector[self._bucket(text.encode("utf-8"))] = 1.0
            return vector
        return [value / norm for value in vector]

    def _bucket(self, data: bytes) -> int:
        digest = hashlib.sha256(data).digest()
        return int.from_bytes(digest[:8], "big") % self._dimension


@dataclass(frozen=True)
class BuiltEmbeddingModel:
    """An embedding object paired with its resolved profile and runtime dim."""

    embeddings: Embeddings
    profile: EmbeddingProfile
    dimension: int


def build_embedding_model(
    config: Config,
    model: str | None = None,
    *,
    error: type[Exception] = VectorIndexError,
) -> BuiltEmbeddingModel:
    """Resolve a profile and build its embedding object.

    For the fake profile the runtime dimension is ``UNI_RAG_EMBEDDING_DIM``. For
    real profiles the optional dependencies are imported lazily and the runtime
    dimension is verified directly from the loaded model.
    """
    profile = resolve_embedding_profile(config, model, error=error)
    if profile.is_fake:
        if profile.dimension <= 0:
            raise error("Embedding dimension must be greater than zero.")
        embeddings: Embeddings = FakeDeterministicEmbeddings(profile.dimension)
        return BuiltEmbeddingModel(
            embeddings=embeddings,
            profile=profile,
            dimension=profile.dimension,
        )

    embeddings = _load_real_embeddings(profile, error=error)
    dimension = _probe_dimension(embeddings, profile, error=error)
    return BuiltEmbeddingModel(
        embeddings=embeddings,
        profile=profile,
        dimension=dimension,
    )


def get_embedding_model(config: Config, model: str | None = None) -> Embeddings:
    """Return the LangChain embedding object for the selected profile.

    Without ``model`` this follows config and returns the deterministic fake
    adapter when ``use_fake_embeddings`` is true. An explicit real ``model``
    overrides the fake default for this call and requires the optional
    ``embeddings`` extra.
    """
    return build_embedding_model(config, model).embeddings


def _load_real_embeddings(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> Embeddings:
    huggingface_embeddings = _require_huggingface(profile, error=error)
    model_kwargs: dict[str, object] = {}
    if profile.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    try:
        return huggingface_embeddings(
            model_name=profile.model_name,
            model_kwargs=model_kwargs,
        )
    except Exception as exc:  # pragma: no cover - requires real model + network
        raise error(
            f"Could not load embedding model '{profile.model_name}': {exc}. "
            f"{profile.access_notes or ''}".strip()
        ) from exc


def _require_huggingface(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> type[Embeddings]:
    """Import ``HuggingFaceEmbeddings`` lazily, mapping failures to ``error``.

    Isolated so tests can monkeypatch it to assert the optional-dependency
    failure path without installing the heavy embeddings extra.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise error(
            f"Embedding model '{profile.model_name}' requires the optional "
            "'embeddings' extra (langchain-huggingface and sentence-transformers). "
            "Install it with: uv sync --extra embeddings"
        ) from exc
    return HuggingFaceEmbeddings


def _probe_dimension(
    embeddings: Embeddings,
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> int:
    try:
        vector = embeddings.embed_query("dimension probe")
    except Exception as exc:  # pragma: no cover - requires real model + network
        raise error(
            f"Could not determine embedding dimension for '{profile.model_name}': {exc}"
        ) from exc
    dimension = len(vector)
    if dimension <= 0:
        raise error(f"Embedding model '{profile.model_name}' returned an empty vector.")
    return dimension
