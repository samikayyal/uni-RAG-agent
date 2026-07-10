"""Embedding model profiles, selection, and physical collection naming.

This module is pure and dependency-light: it never imports ``langchain`` or
``chromadb``. It resolves which embedding profile a command should use (fake vs
a known real Hugging Face profile) and derives the model-namespaced ChromaDB
collection name. Building the actual embedding object lives in ``embeddings.py``
so the heavy/optional dependencies stay lazy.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from uni_rag_agent.config import Config

from .models import VectorIndexError

#: Names that always select the deterministic offline fake adapter.
FAKE_MODEL_NAMES = frozenset({"fake", "fake-embedding"})

DEFAULT_DISTANCE_METRIC = "cosine"


@dataclass(frozen=True)
class EmbeddingProfile:
    """Static description of one embedding model.

    For real profiles ``dimension`` is the documented/declared dimension. The
    actual runtime dimension is verified from the loaded model when a real
    profile is built; the fake profile's declared dimension always equals the
    configured ``UNI_RAG_EMBEDDING_DIM``.
    """

    model_name: str
    provider: str
    dimension: int
    metric: str = DEFAULT_DISTANCE_METRIC
    is_fake: bool = False
    trust_remote_code: bool = False
    gated: bool = False
    requires_extra: str | None = None
    access_notes: str | None = None


#: Known real Hugging Face local-model profiles. These are registry entries
#: only; their optional dependencies (the ``embeddings`` extra) and model
#: weights are not required unless a real profile is actually selected and run.
REAL_EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "BAAI/bge-m3": EmbeddingProfile(
        model_name="BAAI/bge-m3",
        provider="huggingface",
        dimension=1024,
        requires_extra="embeddings",
        access_notes=(
            "Open Sentence Transformers model. The first run downloads weights "
            "from Hugging Face; no access token is required."
        ),
    ),
    "jinaai/jina-embeddings-v3": EmbeddingProfile(
        model_name="jinaai/jina-embeddings-v3",
        provider="huggingface",
        dimension=1024,
        trust_remote_code=True,
        requires_extra="embeddings",
        access_notes=(
            "Requires trust_remote_code=True because it loads custom modeling "
            "code from Hugging Face. Review the remote code before enabling."
        ),
    ),
    "jinaai/jina-embeddings-v5-text-small": EmbeddingProfile(
        model_name="jinaai/jina-embeddings-v5-text-small",
        provider="huggingface",
        dimension=1024,
        trust_remote_code=True,
        gated=True,
        requires_extra="embeddings",
        access_notes=(
            "May be gated and may require trust_remote_code=True. The declared "
            "dimension is provisional: confirm the dimension and access terms "
            "against the model card before production use."
        ),
    ),
    "google/embeddinggemma-300m": EmbeddingProfile(
        model_name="google/embeddinggemma-300m",
        provider="huggingface",
        dimension=768,
        gated=True,
        requires_extra="embeddings",
        access_notes=(
            "Gated model. Accept the license on Hugging Face and authenticate "
            "with a Hugging Face token before the first download."
        ),
    ),
}


def fake_profile(config: Config) -> EmbeddingProfile:
    """Return the deterministic fake profile sized by ``UNI_RAG_EMBEDDING_DIM``."""
    return EmbeddingProfile(
        # Keep fake vectors in their own stable identity even if a user has
        # configured a real model name while experimenting. Otherwise a fake
        # mapping can suppress a later real-model run for that same name.
        model_name="fake-embedding",
        provider="fake",
        dimension=config.embedding_dim,
        is_fake=True,
    )


def resolve_embedding_profile(
    config: Config,
    model: str | None = None,
    *,
    error: type[Exception] = VectorIndexError,
) -> EmbeddingProfile:
    """Choose the embedding profile for a command.

    Selection rules (mirroring the Feature 07 contract):

    * An explicit ``model`` of ``fake``/``fake-embedding`` always selects the
      fake adapter.
    * An explicit real ``model`` selects that real profile and overrides
      ``UNI_RAG_USE_FAKE_EMBEDDINGS=true`` for this command only.
    * Without ``model``, follow config: fake when ``use_fake_embeddings`` is
      true, otherwise the configured ``embedding_model`` must resolve to a known
      real profile or this fails clearly with ``error``.
    """
    requested = model.strip() if model else None

    if requested:
        if requested.casefold() in FAKE_MODEL_NAMES:
            return fake_profile(config)
        profile = REAL_EMBEDDING_PROFILES.get(requested)
        if profile is not None:
            return profile
        raise error(
            f"Unknown embedding model '{requested}'. Known real profiles: "
            f"{', '.join(sorted(REAL_EMBEDDING_PROFILES))}. "
            "Use 'fake-embedding' for offline runs."
        )

    if config.use_fake_embeddings:
        return fake_profile(config)

    profile = REAL_EMBEDDING_PROFILES.get(config.embedding_model)
    if profile is not None:
        return profile
    raise error(
        "Fake embeddings are disabled but the configured embedding model "
        f"'{config.embedding_model}' (provider '{config.embedding_provider}') is "
        "not a known real profile. Set UNI_RAG_USE_FAKE_EMBEDDINGS=true, choose a "
        f"known model ({', '.join(sorted(REAL_EMBEDDING_PROFILES))}), or pass "
        "--model."
    )


def model_slug(model_name: str) -> str:
    """Slugify a model name for use inside a ChromaDB collection name."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-")
    return slug or "model"


def physical_collection_name(
    logical_index: str,
    *,
    provider: str,
    model_name: str,
    dimension: int,
    metric: str = DEFAULT_DISTANCE_METRIC,
) -> str:
    """Return the model-namespaced physical ChromaDB collection name.

    The public logical collection (``document_index`` etc.) stays stable while
    different embedding models/profiles persist into distinct physical
    collections, enabling side-by-side models. The hash input includes
    provider, model, dimension, and metric so an incompatible change rolls over
    to a fresh collection instead of mixing vector spaces.
    """
    digest_input = f"{provider}|{model_name}|{dimension}|{metric}"
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:10]
    return f"{logical_index}__{model_slug(model_name)}__{digest}"
