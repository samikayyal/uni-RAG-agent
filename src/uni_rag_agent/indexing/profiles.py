"""Embedding model profiles, selection, and physical collection naming.

This module is intentionally dependency-light. It contains the canonical
profile registry and never imports a provider SDK. Provider aliases are
resolved here so every downstream caller, including collection naming and
SQLite telemetry, sees the canonical model identity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from uni_rag_agent.config import Config

from .models import VectorIndexError

DEFAULT_DISTANCE_METRIC = "cosine"


@dataclass(frozen=True)
class EmbeddingProfile:
    """Static description of one embedding model.

    ``dimension`` is the declared provider dimension. Local Hugging Face
    models still report their runtime dimension after loading; hosted
    providers use this value as the requested and expected API dimension.
    ``model_name`` is always the canonical registry key.
    """

    model_name: str
    provider: str
    dimension: int
    metric: str = DEFAULT_DISTANCE_METRIC
    trust_remote_code: bool = False
    gated: bool = False
    requires_extra: str | None = None
    access_notes: str | None = None
    aliases: tuple[str, ...] = ()
    api_model_name: str | None = None

    @property
    def declared_dimension(self) -> int:
        """Compatibility/readability alias for the declared dimension."""
        return self.dimension

    @property
    def effective_dimension(self) -> int:
        """The configured provider dimension before local runtime probing."""
        return self.dimension

    @property
    def required_extra(self) -> str | None:
        """Readability alias for the optional dependency extra."""
        return self.requires_extra

    @property
    def effective_api_model_name(self) -> str:
        """Return the provider-facing model identifier."""
        return self.api_model_name or self.model_name

    @property
    def api_model(self) -> str:
        """Short compatibility alias for the provider-facing model name."""
        return self.effective_api_model_name


#: Reviewed Hugging Face local-model profiles. These are registry entries
#: only; their optional dependencies (the ``embeddings`` extra) and model
#: weights are not required unless a real profile is actually selected and run.
EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "BAAI/bge-m3": EmbeddingProfile(
        model_name="BAAI/bge-m3",
        provider="huggingface",
        dimension=1024,
        requires_extra="embeddings",
        api_model_name="BAAI/bge-m3",
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
        api_model_name="jinaai/jina-embeddings-v3",
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
        api_model_name="jinaai/jina-embeddings-v5-text-small",
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
        api_model_name="google/embeddinggemma-300m",
        access_notes=(
            "Gated model. Accept the license on Hugging Face and authenticate "
            "with a Hugging Face token before the first download."
        ),
    ),
    "google/gemini-embedding-001": EmbeddingProfile(
        model_name="google/gemini-embedding-001",
        provider="google_genai",
        dimension=3072,
        requires_extra="embeddings-cloud",
        api_model_name="gemini-embedding-001",
        aliases=("gemini-embedding-001",),
        access_notes=(
            "Uses the direct Gemini Developer API and requires GOOGLE_API_KEY. "
            "Vertex AI is not used by this profile."
        ),
    ),
    "Qwen/Qwen3-Embedding-8B": EmbeddingProfile(
        model_name="Qwen/Qwen3-Embedding-8B",
        provider="nebius",
        dimension=4096,
        requires_extra="embeddings-cloud",
        api_model_name="Qwen/Qwen3-Embedding-8B",
        access_notes=(
            "Uses the Nebius Token Factory OpenAI-compatible API and requires "
            "NEBIUS_API_KEY."
        ),
    ),
}


def _build_alias_index() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, profile in EMBEDDING_PROFILES.items():
        for alias in profile.aliases:
            previous = aliases.setdefault(alias, canonical)
            if previous != canonical:  # pragma: no cover - registry invariant
                raise RuntimeError(
                    f"Embedding model alias '{alias}' is registered more than once."
                )
    return aliases


_EMBEDDING_ALIASES = _build_alias_index()


def resolve_embedding_profile(
    config: Config,
    model: str | None = None,
    *,
    error: type[Exception] = VectorIndexError,
) -> EmbeddingProfile:
    """Resolve an explicit or configured model to a reviewed profile."""
    requested = model.strip() if model and model.strip() else None
    configured = getattr(config, "embedding_model", None)
    if requested is None and configured:
        requested = configured.strip() or None
    if not requested:
        raise error(
            "No embedding model selected. Set UNI_RAG_EMBEDDING_MODEL or pass "
            f"--model. Supported profiles: {', '.join(sorted(EMBEDDING_PROFILES))}."
        )

    canonical = requested
    if canonical not in EMBEDDING_PROFILES:
        canonical = _EMBEDDING_ALIASES.get(requested)
    profile = EMBEDDING_PROFILES.get(canonical) if canonical else None
    if profile is not None:
        return profile
    raise error(
        f"Unknown embedding model '{requested}'. Supported profiles: "
        f"{', '.join(sorted(EMBEDDING_PROFILES))}."
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
