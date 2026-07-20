"""Lazy direct-Gemini embedding provider."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from uni_rag_agent.config import Config

from ..profiles import EmbeddingProfile
from .common import (
    EmbeddingValidationError,
    retry_transient,
    sanitize_provider_error,
    validate_vectors,
)

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"

GEMINI_BATCH_POLL_SECONDS = 10.0

_GEMINI_BATCH_SUCCEEDED = "JOB_STATE_SUCCEEDED"
_GEMINI_BATCH_TERMINAL_STATES = frozenset(
    {
        _GEMINI_BATCH_SUCCEEDED,
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }
)


class GoogleGenAIEmbeddings(Embeddings):
    """Validated Gemini adapter using batch jobs for document indexing."""

    def __init__(self, client: object, *, model: str, dimension: int) -> None:
        self.client = client
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        try:
            # Batch creation is not idempotent. Retrying an ambiguous failure can
            # submit the same paid work twice, so only polling reads are retried.
            job = self.client.batches.create_embeddings(  # type: ignore[attr-defined]
                model=self.model,
                src={
                    "inlined_requests": {
                        "contents": [{"parts": [{"text": text}]} for text in values],
                        "config": {
                            "task_type": DOCUMENT_TASK,
                            "output_dimensionality": self.dimension,
                        },
                    }
                },
            )
            state = _batch_state_name(job)
            while state not in _GEMINI_BATCH_TERMINAL_STATES:
                time.sleep(GEMINI_BATCH_POLL_SECONDS)
                job = retry_transient(
                    lambda: self.client.batches.get(  # type: ignore[attr-defined]
                        name=job.name
                    ),
                    provider="Google GenAI",
                )
                state = _batch_state_name(job)
            if state != _GEMINI_BATCH_SUCCEEDED:
                raise RuntimeError("Google GenAI batch job did not succeed")
            responses = getattr(
                getattr(job, "dest", None),
                "inlined_embed_content_responses",
                None,
            )
            vectors = [
                item.response.embedding.values
                for item in responses or []
                if getattr(item, "response", None)
                and getattr(item.response, "embedding", None)
            ]
            return validate_vectors(
                vectors,
                expected_count=len(values),
                expected_dimension=self.dimension,
                context="Google GenAI embedding response",
            )
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Google GenAI returned an invalid embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Google GenAI",
                    operation="document embedding",
                    model=self.model,
                )
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed retrieval queries in one synchronous Gemini batch request."""
        values = list(texts)
        if not values:
            return []
        try:
            result = retry_transient(
                lambda: self.client.models.embed_content(  # type: ignore[attr-defined]
                    model=self.model,
                    contents=[{"parts": [{"text": text}]} for text in values],
                    config={
                        "task_type": QUERY_TASK,
                        "output_dimensionality": self.dimension,
                    },
                ),
                provider="Google GenAI",
            )
            vectors = [embedding.values for embedding in result.embeddings or []]
            return validate_vectors(
                vectors,
                expected_count=len(values),
                expected_dimension=self.dimension,
                context="Google GenAI query embedding response",
            )
        except EmbeddingValidationError as exc:
            raise RuntimeError(
                "Google GenAI returned an invalid query embedding response."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                sanitize_provider_error(
                    exc,
                    "Google GenAI",
                    operation="query embedding",
                    model=self.model,
                )
            ) from exc


def build_embeddings(
    config: Config,
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> tuple[GoogleGenAIEmbeddings, int]:
    """Construct direct Gemini embeddings without a dimension-probe call."""
    api_key = _google_api_key(config)
    if not api_key:
        raise error(
            "Google GenAI embedding provider requires GOOGLE_API_KEY. "
            "Set it in the merged .env file or environment."
        )

    constructor = _require_google_genai(profile, error=error)
    dimension = profile.dimension
    try:
        client = constructor(api_key=api_key, vertexai=False)
    except Exception as exc:  # pragma: no cover - provider construction boundary
        detail = sanitize_provider_error(
            exc,
            "Google GenAI",
            operation="construction",
            model=profile.effective_api_model_name,
        )
        if profile.access_notes:
            detail = f"{detail} {profile.access_notes}"
        raise error(detail) from exc

    return (
        GoogleGenAIEmbeddings(
            client,
            model=profile.effective_api_model_name,
            dimension=dimension,
        ),
        dimension,
    )


def _google_api_key(config: Config) -> str | None:
    configured = getattr(config, "google_api_key", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    value = os.getenv("GOOGLE_API_KEY")
    return value.strip() if value and value.strip() else None


def _batch_state_name(job: object) -> str:
    """Return an SDK batch state without importing the optional SDK eagerly."""
    state = getattr(job, "state", None)
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(state, str):
        return state
    raise EmbeddingValidationError("Google GenAI batch job has no valid state.")


def _require_google_genai(
    profile: EmbeddingProfile,
    *,
    error: type[Exception],
) -> type[object]:
    """Import the Google GenAI client only for the hosted profile."""
    extra = profile.requires_extra or "embeddings-cloud"
    try:
        from google import genai
    except ImportError as exc:
        raise error(
            "Google GenAI embedding provider requires the optional "
            f"'{extra}' extra (google-genai). "
            f"Install it with: uv sync --extra {extra}"
        ) from exc
    return genai.Client


build_google_genai_embeddings = build_embeddings
