from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from uni_rag_agent.indexing import (
    EMBEDDING_PROFILES,
    SemanticSearchError,
    VectorIndexError,
    build_embedding_model,
    physical_collection_name,
    resolve_embedding_profile,
    semantic_search,
    semantic_search_many,
    sync_vector_index,
)
from uni_rag_agent.indexing.embedding_providers import factory as factory_module
from uni_rag_agent.indexing.embedding_providers import (
    google_genai as google_genai_module,
)
from uni_rag_agent.indexing.embedding_providers import huggingface as huggingface_module
from uni_rag_agent.indexing.embedding_providers import nebius as nebius_module
from uni_rag_agent.indexing import vector as vector_module
from uni_rag_agent.retrieval import RetrievalResult
from uni_rag_agent import cli as cli_module
from uni_rag_agent.storage import connect_sqlite
from tests.sqlite_helpers import insert_minimal_chunk
from tests.embedding_doubles import (
    DeterministicTestEmbeddings,
    TEST_EMBEDDING_DIMENSIONS,
    embeddings_for_model,
)
from tests.support import (
    clean_subprocess_env,
    initialized_connection,
    make_config as make_test_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path: Path, **overrides: object):
    overrides.setdefault("embedding_model", "BAAI/bge-m3")
    return make_test_config(tmp_path, **overrides)


@pytest.fixture()
def patch_huggingface_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_test_embeddings(**kwargs: object) -> DeterministicTestEmbeddings:
        return embeddings_for_model(str(kwargs["model_name"]))

    monkeypatch.setattr(
        huggingface_module,
        "_require_huggingface",
        lambda *_args, **_kwargs: load_test_embeddings,
    )


class _GoogleHostedClient:
    def __init__(self, dimension: int, recorder: dict[str, object]) -> None:
        self._embeddings = DeterministicTestEmbeddings(dimension)
        self._recorder = recorder
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> list[list[float]]:
        self.document_calls += 1
        self._recorder.setdefault("google_document_kwargs", []).append(kwargs)  # type: ignore[union-attr]
        return self._embeddings.embed_documents(list(texts))

    def embed_query(self, text: str, **kwargs: object) -> list[float]:
        self.query_calls += 1
        self._recorder.setdefault("google_query_kwargs", []).append(kwargs)  # type: ignore[union-attr]
        return self._embeddings.embed_query(text)


class _NebiusHostedClient:
    def __init__(self, dimension: int, recorder: dict[str, object]) -> None:
        self._embeddings = DeterministicTestEmbeddings(dimension)
        self._recorder = recorder
        self.calls = 0

    def create(
        self,
        *,
        model: str,
        input: str | list[str],
        dimensions: int,
    ) -> SimpleNamespace:
        self.calls += 1
        self._recorder.setdefault("nebius_request_kwargs", []).append(  # type: ignore[union-attr]
            {"model": model, "input": input, "dimensions": dimensions}
        )
        values = [input] if isinstance(input, str) else list(input)
        vectors = self._embeddings.embed_documents(values)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(vectors)
            ]
        )


@pytest.fixture()
def patch_hosted_provider_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Inject deterministic clients through the live hosted provider seams."""
    recorder: dict[str, object] = {}

    def google_constructor(**kwargs: object) -> _GoogleHostedClient:
        recorder["google_constructor_kwargs"] = kwargs
        client = _GoogleHostedClient(3072, recorder)
        recorder["google_client"] = client
        return client

    def nebius_constructor(**kwargs: object) -> SimpleNamespace:
        recorder["nebius_constructor_kwargs"] = kwargs
        client = _NebiusHostedClient(4096, recorder)
        recorder["nebius_client"] = client
        return SimpleNamespace(embeddings=client)

    monkeypatch.setattr(
        google_genai_module,
        "_google_api_key",
        lambda _config: "offline-google-test-key",
    )
    monkeypatch.setattr(
        google_genai_module,
        "_require_google_embeddings",
        lambda *_args, **_kwargs: google_constructor,
    )
    monkeypatch.setattr(
        google_genai_module,
        "_require_google_genai",
        lambda *_args, **_kwargs: google_constructor,
        raising=False,
    )
    monkeypatch.setattr(
        nebius_module,
        "_nebius_api_key",
        lambda _config: "offline-nebius-test-key",
    )
    monkeypatch.setattr(
        nebius_module,
        "_require_openai",
        lambda *_args, **_kwargs: nebius_constructor,
    )
    monkeypatch.setattr(
        nebius_module,
        "_require_nebius_openai",
        lambda *_args, **_kwargs: nebius_constructor,
        raising=False,
    )
    return recorder


# --------------------------------------------------------------------------- #
# Embedding profile registry (pure, offline)
# --------------------------------------------------------------------------- #


def test_resolve_profile_follows_configured_model(tmp_path: Path) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")

    profile = resolve_embedding_profile(config)

    assert profile.model_name == "BAAI/bge-m3"
    assert profile.provider == "huggingface"
    assert profile.dimension > 0


def test_explicit_model_overrides_unset_configuration(tmp_path: Path) -> None:
    config = make_config(tmp_path, embedding_model=None)
    profile = resolve_embedding_profile(config, "BAAI/bge-m3")

    assert profile.model_name == "BAAI/bge-m3"


def test_missing_model_fails_with_supported_profiles(tmp_path: Path) -> None:
    config = make_config(tmp_path, embedding_model=None)
    with pytest.raises(VectorIndexError, match="UNI_RAG_EMBEDDING_MODEL"):
        resolve_embedding_profile(config)
    with pytest.raises(SemanticSearchError, match="Supported profiles"):
        resolve_embedding_profile(config, error=SemanticSearchError)


def test_unknown_model_fails_clearly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(VectorIndexError, match="Unknown embedding model"):
        resolve_embedding_profile(config, "no/such-model")


def test_unknown_configured_model_fails_generically(tmp_path: Path) -> None:
    config = make_config(tmp_path, embedding_model="no/such-model")
    with pytest.raises(SemanticSearchError, match="Unknown embedding model"):
        resolve_embedding_profile(config, error=SemanticSearchError)


def test_real_profile_registry_metadata() -> None:
    assert set(EMBEDDING_PROFILES) == {
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3",
        "jinaai/jina-embeddings-v5-text-small",
        "google/embeddinggemma-300m",
        "google/gemini-embedding-001",
        "Qwen/Qwen3-Embedding-8B",
    }
    local_profiles = {
        profile
        for profile in EMBEDDING_PROFILES.values()
        if profile.provider == "huggingface"
    }
    assert len(local_profiles) == 4
    for profile in local_profiles:
        assert profile.provider == "huggingface"
        assert profile.requires_extra == "embeddings"
        assert profile.metric == "cosine"
        assert profile.access_notes

    hosted_profiles = {
        profile.model_name: profile
        for profile in EMBEDDING_PROFILES.values()
        if profile.provider != "huggingface"
    }
    assert hosted_profiles["google/gemini-embedding-001"].provider == "google_genai"
    assert hosted_profiles["google/gemini-embedding-001"].requires_extra == (
        "embeddings-cloud"
    )
    assert hosted_profiles["google/gemini-embedding-001"].dimension == 3072
    assert hosted_profiles["google/gemini-embedding-001"].aliases == (
        "gemini-embedding-001",
    )
    assert hosted_profiles["Qwen/Qwen3-Embedding-8B"].provider == "nebius"
    assert hosted_profiles["Qwen/Qwen3-Embedding-8B"].requires_extra == (
        "embeddings-cloud"
    )
    assert hosted_profiles["Qwen/Qwen3-Embedding-8B"].dimension == 4096

    assert EMBEDDING_PROFILES["jinaai/jina-embeddings-v3"].trust_remote_code is True
    assert EMBEDDING_PROFILES["BAAI/bge-m3"].dimension == 1024
    gemma = EMBEDDING_PROFILES["google/embeddinggemma-300m"]
    assert gemma.gated is True
    assert gemma.dimension == 768


# --------------------------------------------------------------------------- #
# Test embedding double (deterministic, offline)
# --------------------------------------------------------------------------- #


def test_test_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = DeterministicTestEmbeddings(16)
    first = embeddings.embed_query("distributed computation")
    second = embeddings.embed_query("distributed computation")

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)

    assert sum(a * b for a, b in zip(first, second)) == pytest.approx(1.0)
    shared = embeddings.embed_query("distributed systems")
    disjoint = embeddings.embed_query("xylophone qwerty")
    shared_similarity = sum(a * b for a, b in zip(first, shared))
    disjoint_similarity = sum(a * b for a, b in zip(first, disjoint))
    assert shared_similarity > disjoint_similarity
    assert disjoint_similarity < 0.5
    assert embeddings.embed_query("")


def test_test_embeddings_return_normalized_vectors_for_token_text() -> None:
    embeddings = DeterministicTestEmbeddings(8)

    vector = embeddings.embed_query("computation overview")

    assert len(vector) == 8
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_test_embedding_dimensions_match_reviewed_profiles() -> None:
    assert set(TEST_EMBEDDING_DIMENSIONS) == set(EMBEDDING_PROFILES)


def test_build_embedding_model_uses_runtime_test_dimension(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    built = build_embedding_model(config)

    assert isinstance(
        built.embeddings,
        huggingface_module.HuggingFaceEmbeddingsAdapter,
    )
    assert isinstance(built.embeddings.client, DeterministicTestEmbeddings)
    assert built.dimension == TEST_EMBEDDING_DIMENSIONS["BAAI/bge-m3"]
    assert (
        len(built.embeddings.embed_query("probe"))
        == TEST_EMBEDDING_DIMENSIONS["BAAI/bge-m3"]
    )


def test_model_loader_forwards_trust_remote_code_to_jina_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_huggingface_loader: None,
) -> None:
    captured: dict[str, object] = {}

    def test_constructor(**kwargs: object) -> DeterministicTestEmbeddings:
        captured.update(kwargs)
        return embeddings_for_model(str(kwargs["model_name"]))

    monkeypatch.setattr(
        huggingface_module,
        "_require_huggingface",
        lambda *_args, **_kwargs: test_constructor,
    )

    build_embedding_model(make_config(tmp_path), "jinaai/jina-embeddings-v3")

    assert captured["model_name"] == "jinaai/jina-embeddings-v3"
    assert captured["model_kwargs"] == {"trust_remote_code": True}


def test_embedding_model_log_label_normalizes_whitespace_override(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")

    assert cli_module._embedding_model_log_label(config, "  ") == "BAAI/bge-m3"
    assert (
        cli_module._embedding_model_log_label(config, "  google/embeddinggemma-300m  ")
        == "google/embeddinggemma-300m"
    )
    assert (
        cli_module._embedding_model_log_label(config, "  gemini-embedding-001  ")
        == "google/gemini-embedding-001"
    )


def test_missing_model_uses_the_callers_domain_error(tmp_path: Path) -> None:
    config = make_config(tmp_path, embedding_model=None)
    with initialized_connection(config):
        pass

    with pytest.raises(VectorIndexError, match="No embedding model selected"):
        sync_vector_index(config)
    with pytest.raises(SemanticSearchError, match="No embedding model selected"):
        semantic_search(config, "distributed")


# --------------------------------------------------------------------------- #
# Optional-dependency failure path (monkeypatched, never loads a real model)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def force_missing_embeddings_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def require_missing_extra(profile: object, *, error: type[Exception]) -> object:
        raise error(
            "Embedding model requires the optional 'embeddings' extra. "
            "Install it with: uv sync --extra embeddings"
        )

    monkeypatch.setattr(
        huggingface_module, "_require_huggingface", require_missing_extra
    )


def test_model_missing_extra_fails_clearly_for_builder(
    tmp_path: Path,
    force_missing_embeddings_extra: None,
) -> None:
    config = make_config(tmp_path)
    with pytest.raises(VectorIndexError, match="uv sync --extra embeddings"):
        build_embedding_model(config, "BAAI/bge-m3")


def test_real_model_missing_extra_fails_clearly_for_sync_and_search(
    tmp_path: Path,
    force_missing_embeddings_extra: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config):
        pass

    with pytest.raises(VectorIndexError, match="embeddings' extra"):
        sync_vector_index(config, model="BAAI/bge-m3")
    with pytest.raises(SemanticSearchError, match="embeddings' extra"):
        semantic_search(config, "distributed", model="BAAI/bge-m3")


# --------------------------------------------------------------------------- #
# Physical collection naming
# --------------------------------------------------------------------------- #


def test_physical_collection_name_is_model_namespaced_and_stable() -> None:
    first = physical_collection_name(
        "document_index",
        provider="huggingface",
        model_name="BAAI/bge-m3",
        dimension=8,
        metric="cosine",
    )
    same = physical_collection_name(
        "document_index",
        provider="huggingface",
        model_name="BAAI/bge-m3",
        dimension=8,
        metric="cosine",
    )
    second = physical_collection_name(
        "document_index",
        provider="huggingface",
        model_name="google/embeddinggemma-300m",
        dimension=16,
        metric="cosine",
    )

    assert first == same
    assert first != second
    assert first.startswith("document_index__baai-bge-m3__")
    assert second.startswith("document_index__google-embeddinggemma-300m__")


# --------------------------------------------------------------------------- #
# Vector sync (test embeddings + ChromaDB)
# --------------------------------------------------------------------------- #


def test_sync_indexes_only_current_eligible_chunks(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        document = insert_minimal_chunk(
            connection,
            config,
            filename="notes.md",
            source_type="document",
            text="distributed computation with MapReduce",
        )
        data_schema = insert_minimal_chunk(
            connection,
            config,
            filename="dataset.csv",
            extension=".csv",
            category="data_schema",
            source_type="data_schema",
            text="column term score",
        )
        insert_minimal_chunk(
            connection,
            config,
            filename="failed.md",
            index_status="failed",
            source_type="document",
            text="failed should not embed",
        )
        insert_minimal_chunk(
            connection,
            config,
            filename="pending.md",
            index_status="pending",
            source_type="document",
            text="pending should not embed",
        )
        insert_minimal_chunk(
            connection,
            config,
            filename="empty.md",
            source_type="document",
            text="   ",
        )
        connection.commit()

    result = sync_vector_index(config)

    assert result.rebuild is False
    assert result.model == "BAAI/bge-m3"
    assert result.provider == "huggingface"
    assert result.chunks_seen == 2
    assert result.vectors_indexed == 2
    assert result.embeddings_total == 2
    assert result.by_source_type == {"data_schema": 1, "document": 1}

    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, vector_backend, vector_collection, vector_id,
                   embedding_model, embedding_dim
            FROM embeddings
            ORDER BY chunk_id
            """
        ).fetchall()

    indexed_ids = {row["chunk_id"] for row in rows}
    assert indexed_ids == {document.chunk_id, data_schema.chunk_id}
    for row in rows:
        assert row["vector_backend"] == "chroma"
        assert row["vector_id"] == f"chunk:{row['chunk_id']}"
        assert row["embedding_model"] == "BAAI/bge-m3"
        assert row["embedding_dim"] == TEST_EMBEDDING_DIMENSIONS["BAAI/bge-m3"]
    document_row = next(r for r in rows if r["chunk_id"] == document.chunk_id)
    assert document_row["vector_collection"].startswith("document_index__")


def test_sync_can_print_embedding_progress(
    tmp_path: Path,
    patch_huggingface_loader: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        insert_minimal_chunk(
            connection,
            config,
            filename="notes.md",
            source_type="document",
            text="distributed computation with MapReduce",
        )
        connection.commit()

    sync_vector_index(config, collection="document_index", show_progress=True)

    assert capsys.readouterr().out.splitlines() == [
        "Embedding document_index: 0/1",
        "Embedding document_index: 1/1",
    ]


@pytest.mark.parametrize(
    ("selected_model", "canonical_model", "provider", "dimension"),
    [
        (
            "gemini-embedding-001",
            "google/gemini-embedding-001",
            "google_genai",
            3072,
        ),
        (
            "Qwen/Qwen3-Embedding-8B",
            "Qwen/Qwen3-Embedding-8B",
            "nebius",
            4096,
        ),
    ],
)
def test_hosted_profiles_flow_through_real_chroma_and_sqlite(
    tmp_path: Path,
    patch_hosted_provider_doubles: dict[str, object],
    selected_model: str,
    canonical_model: str,
    provider: str,
    dimension: int,
) -> None:
    config = make_config(
        tmp_path,
        embedding_model=selected_model,
        google_api_key="offline-google-test-key",
        nebius_api_key="offline-nebius-test-key",
    )
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            filename="hosted.md",
            text="distributed computation with hosted embeddings",
        )
        connection.commit()

    result = sync_vector_index(
        config, model=selected_model, collection="document_index"
    )

    assert result.model == canonical_model
    assert result.provider == provider
    assert result.embedding_dim == dimension
    assert result.vectors_indexed == 1

    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            """
            SELECT embedding_model, embedding_dim, vector_collection, vector_id
            FROM embeddings
            WHERE chunk_id = ?
            """,
            (stored.chunk_id,),
        ).fetchone()
    assert row is not None
    assert row["embedding_model"] == canonical_model
    assert row["embedding_dim"] == dimension
    assert row["vector_id"] == f"chunk:{stored.chunk_id}"
    assert row["vector_collection"] == physical_collection_name(
        "document_index",
        provider=provider,
        model_name=canonical_model,
        dimension=dimension,
        metric="cosine",
    )

    results = semantic_search(
        config,
        "distributed computation",
        indexes=("document_index",),
        model=selected_model,
    )
    assert [item.chunk_id for item in results] == [stored.chunk_id]


def test_gemini_alias_and_canonical_selection_share_one_physical_profile(
    tmp_path: Path,
    patch_hosted_provider_doubles: dict[str, object],
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="gemini-embedding-001",
        google_api_key="offline-google-test-key",
    )
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            filename="alias.md",
            text="canonical Gemini collection identity",
        )
        connection.commit()

    alias_result = sync_vector_index(config, model="gemini-embedding-001")
    canonical_result = sync_vector_index(
        config,
        model="google/gemini-embedding-001",
    )

    assert alias_result.model == canonical_result.model == "google/gemini-embedding-001"
    assert alias_result.vectors_indexed == 1
    assert canonical_result.vectors_indexed == 0
    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute(
            """
            SELECT embedding_model, vector_collection
            FROM embeddings
            WHERE chunk_id = ?
            """,
            (stored.chunk_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["embedding_model"] == "google/gemini-embedding-001"


def test_exhausted_hosted_retries_preserve_completed_batches_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="google/gemini-embedding-001",
        google_api_key="offline-google-test-key",
    )
    with initialized_connection(config) as connection:
        for index in range(65):
            insert_minimal_chunk(
                connection,
                config,
                filename=f"batch-{index}.md",
                text=f"batch document {index} distributed computation",
            )
        connection.commit()

    class RateLimitError(Exception):
        status_code = 429

    class FailingGoogleClient:
        def __init__(self, *, fail_single: bool) -> None:
            self._embeddings = DeterministicTestEmbeddings(3072)
            self.fail_single = fail_single
            self.calls = 0

        def embed_documents(
            self, texts: list[str], **_kwargs: object
        ) -> list[list[float]]:
            self.calls += 1
            if self.fail_single and len(texts) == 1:
                raise RateLimitError("secret-key must not surface")
            return self._embeddings.embed_documents(texts)

        def embed_query(self, text: str, **_kwargs: object) -> list[float]:
            return self._embeddings.embed_query(text)

    failing_client = FailingGoogleClient(fail_single=True)
    monkeypatch.setattr(
        google_genai_module,
        "_google_api_key",
        lambda _config: "offline-google-test-key",
    )
    monkeypatch.setattr(
        google_genai_module,
        "_require_google_embeddings",
        lambda *_args, **_kwargs: lambda **_kwargs: failing_client,
    )

    with pytest.raises(
        VectorIndexError, match="Google GenAI embedding document embedding failed"
    ):
        sync_vector_index(config, collection="document_index")

    with closing(connect_sqlite(config)) as connection:
        committed = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert committed == 64
    assert failing_client.calls == 4  # one successful batch plus three retries

    successful_client = FailingGoogleClient(fail_single=False)
    monkeypatch.setattr(
        google_genai_module,
        "_require_google_embeddings",
        lambda *_args, **_kwargs: lambda **_kwargs: successful_client,
    )
    resumed = sync_vector_index(config, collection="document_index")

    assert resumed.vectors_indexed == 1
    with closing(connect_sqlite(config)) as connection:
        total = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert total == 65


def test_sync_is_idempotent(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        insert_minimal_chunk(connection, config, filename="a.md", text="alpha beta")
        insert_minimal_chunk(connection, config, filename="b.md", text="gamma delta")
        connection.commit()

    first = sync_vector_index(config)
    second = sync_vector_index(config)

    assert first.vectors_indexed == 2
    assert second.vectors_indexed == 0
    assert second.embeddings_total == 2
    assert any("already embedded" in diagnostic for diagnostic in second.diagnostics)
    with closing(connect_sqlite(config)) as connection:
        total = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert total == 2


def test_sync_rolls_over_to_a_second_profile_and_rebuilds_it(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            filename="rollover.md",
            text="distributed computation",
        )
        connection.commit()

    first = sync_vector_index(config)
    changed = dataclasses.replace(
        config,
        embedding_model="google/embeddinggemma-300m",
    )
    rollover = sync_vector_index(changed)
    rebuilt = sync_vector_index(changed, rebuild=True)

    assert first.vectors_indexed == 1
    assert rollover.vectors_indexed == 1
    assert rebuilt.rows_removed == 1
    assert rebuilt.vectors_indexed == 1
    assert [result.chunk_id for result in semantic_search(changed, "distributed")] == [
        stored.chunk_id
    ]
    with closing(connect_sqlite(changed)) as connection:
        rows = connection.execute(
            "SELECT vector_collection FROM embeddings WHERE chunk_id = ?",
            (stored.chunk_id,),
        ).fetchall()
    assert len(rows) == 2
    assert len({row["vector_collection"] for row in rows}) == 2


def test_two_profiles_create_distinct_mappings(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            filename="profile.md",
            text="distributed computation",
        )
        connection.commit()

    first_result = sync_vector_index(config)
    second_result = sync_vector_index(
        config,
        model="google/embeddinggemma-300m",
    )

    assert first_result.model == "BAAI/bge-m3"
    assert second_result.model == "google/embeddinggemma-300m"
    assert first_result.embedding_dim == TEST_EMBEDDING_DIMENSIONS["BAAI/bge-m3"]
    assert (
        second_result.embedding_dim
        == TEST_EMBEDDING_DIMENSIONS["google/embeddinggemma-300m"]
    )
    assert first_result.vectors_indexed == 1
    assert second_result.vectors_indexed == 1
    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute(
            """
            SELECT embedding_model, vector_collection
            FROM embeddings
            WHERE chunk_id = ?
            ORDER BY embedding_model
            """,
            (stored.chunk_id,),
        ).fetchall()
    assert [row["embedding_model"] for row in rows] == [
        "BAAI/bge-m3",
        "google/embeddinggemma-300m",
    ]
    assert len({row["vector_collection"] for row in rows}) == 2


def test_sync_collection_filter_limits_to_one_logical_index(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        document = insert_minimal_chunk(
            connection, config, filename="notes.md", source_type="document", text="doc"
        )
        insert_minimal_chunk(
            connection,
            config,
            filename="slides.pptx",
            extension=".pptx",
            category="slides",
            source_type="slides",
            text="slide text",
        )
        connection.commit()

    result = sync_vector_index(config, collection="document_index")

    assert result.collections == ("document_index",)
    assert result.vectors_indexed == 1
    assert result.by_source_type == {"document": 1}
    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute("SELECT chunk_id FROM embeddings").fetchall()
    assert [row["chunk_id"] for row in rows] == [document.chunk_id]


def test_sync_unknown_collection_fails_clearly(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config):
        pass
    with pytest.raises(VectorIndexError, match="Unknown logical index"):
        sync_vector_index(config, collection="slides")


def test_rebuild_removes_stale_rows_and_repopulates_selected_model(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        keep = insert_minimal_chunk(
            connection, config, filename="keep.md", text="alpha beta"
        )
        drop = insert_minimal_chunk(
            connection, config, filename="drop.md", text="gamma delta"
        )
        connection.commit()

    first = sync_vector_index(config)
    assert first.vectors_indexed == 2

    with closing(connect_sqlite(config)) as connection:
        connection.execute(
            "UPDATE files SET index_status = 'failed' WHERE id = ?", (drop.file_id,)
        )
        connection.commit()

    rebuilt = sync_vector_index(config, rebuild=True)

    assert rebuilt.rebuild is True
    assert rebuilt.rows_removed == 2
    assert rebuilt.vectors_indexed == 1
    assert rebuilt.embeddings_total == 1
    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute("SELECT chunk_id FROM embeddings").fetchall()
    assert [row["chunk_id"] for row in rows] == [keep.chunk_id]


def test_sync_reconciles_orphaned_vectors_and_reused_chunk_ids(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        old = insert_minimal_chunk(
            connection,
            config,
            course_name="Old Course",
            filename="old.md",
            text="distributed computation",
        )
        connection.commit()
    sync_vector_index(config)

    with closing(connect_sqlite(config)) as connection:
        connection.execute("DELETE FROM chunks WHERE id = ?", (old.chunk_id,))
        replacement = insert_minimal_chunk(
            connection,
            config,
            course_name="New Course",
            filename="replacement.md",
            text="xylophone qwerty",
        )
        connection.commit()

    assert replacement.chunk_id == old.chunk_id
    # The stale Chroma vector has the same id, but no longer has an exact
    # authoritative SQLite mapping and must not hydrate as the replacement.
    assert semantic_search(config, "distributed computation") == []

    repaired = sync_vector_index(config)

    assert repaired.vectors_removed == 1
    assert repaired.vectors_indexed == 1
    assert [result.chunk_id for result in semantic_search(config, "xylophone")] == [
        replacement.chunk_id
    ]


def test_sync_restores_a_missing_physical_collection(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            filename="missing-collection.md",
            text="distributed computation",
        )
        connection.commit()
    sync_vector_index(config)

    with closing(connect_sqlite(config)) as connection:
        physical = connection.execute(
            "SELECT vector_collection FROM embeddings WHERE chunk_id = ?",
            (stored.chunk_id,),
        ).fetchone()[0]
    client = vector_module._chroma_client(config, error=VectorIndexError)
    client.delete_collection(name=physical)  # type: ignore[attr-defined]

    repaired = sync_vector_index(config)
    stable = sync_vector_index(config)

    assert repaired.mappings_removed == 1
    assert repaired.vectors_indexed == 1
    assert stable.mappings_removed == 0
    assert stable.vectors_removed == 0
    assert stable.vectors_indexed == 0
    assert [result.chunk_id for result in semantic_search(config, "distributed")] == [
        stored.chunk_id
    ]


def test_sync_restores_a_missing_individual_vector(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            filename="missing-vector.md",
            text="distributed computation",
        )
        connection.commit()
    sync_vector_index(config)

    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            """
            SELECT vector_collection, vector_id
            FROM embeddings
            WHERE chunk_id = ?
            """,
            (stored.chunk_id,),
        ).fetchone()
    client = vector_module._chroma_client(config, error=VectorIndexError)
    collection = client.get_collection(name=row["vector_collection"])  # type: ignore[attr-defined]
    collection.delete(ids=[row["vector_id"]])  # type: ignore[attr-defined]

    repaired = sync_vector_index(config)

    assert repaired.mappings_removed == 1
    assert repaired.vectors_indexed == 1
    assert [result.chunk_id for result in semantic_search(config, "distributed")] == [
        stored.chunk_id
    ]


def test_sync_reports_no_eligible_chunks(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config):
        pass

    result = sync_vector_index(config)

    assert result.vectors_indexed == 0
    assert result.chunks_seen == 0
    assert any("No eligible indexed chunks" in d for d in result.diagnostics)


# --------------------------------------------------------------------------- #
# Semantic search (test embeddings)
# --------------------------------------------------------------------------- #


def test_semantic_search_many_batches_queries_and_chroma_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    profile = resolve_embedding_profile(config, error=SemanticSearchError)
    dimension = 3
    physical = vector_module._physical_name("document_index", profile, dimension)

    class FakeEmbeddings:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_queries(self, queries: list[str]) -> list[list[float]]:
            self.calls.append(queries)
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    class FakeCollection:
        def __init__(self) -> None:
            self.query_calls = 0

        def query(self, **kwargs: object) -> dict[str, object]:
            self.query_calls += 1
            assert len(kwargs["query_embeddings"]) == 2
            return {
                "ids": [["chunk:1"], ["chunk:2"]],
                "distances": [[0.1], [0.2]],
                "metadatas": [[{"chunk_id": 1}], [{"chunk_id": 2}]],
            }

    embeddings = FakeEmbeddings()
    collection = FakeCollection()
    context = SimpleNamespace(
        built=SimpleNamespace(
            embeddings=embeddings,
            profile=profile,
            dimension=dimension,
        ),
        collections={physical: collection},
        counts={physical: 2},
    )
    monkeypatch.setattr(
        vector_module,
        "_build_semantic_context",
        lambda *args, **kwargs: context,
    )
    rows = {
        1: {
            "chunk_id": 1,
            "file_id": 1,
            "course": "Information Retrieval",
            "file_path": "notes.md",
            "source_type": "document",
            "location_type": None,
            "location_value": None,
            "text": "first",
        },
        2: {
            "chunk_id": 2,
            "file_id": 1,
            "course": "Information Retrieval",
            "file_path": "notes.md",
            "source_type": "document",
            "location_type": None,
            "location_value": None,
            "text": "second",
        },
    }
    monkeypatch.setattr(
        vector_module,
        "_hydrate_candidates",
        lambda _config, *, candidates, **kwargs: [
            rows[chunk_id] for chunk_id in candidates
        ],
    )

    result_sets = semantic_search_many(
        config,
        ["first query", "second query"],
        indexes=["document_index"],
        top_k=1,
    )

    assert [results[0].chunk_id for results in result_sets] == [1, 2]
    assert embeddings.calls == [["first query", "second query"]]
    assert collection.query_calls == 1


def test_semantic_search_returns_sqlite_joined_results(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        relevant = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="syllabus.txt",
            extension=".txt",
            source_type="document",
            text="distributed computation with MapReduce",
            location_type="page",
            location_value="3",
        )
        insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="other.md",
            text="neural networks gradient descent",
        )
        connection.commit()
    sync_vector_index(config)

    results = semantic_search(config, "distributed computation with MapReduce")

    assert results[0].chunk_id == relevant.chunk_id
    assert results[0].retrieval_method == "semantic"
    assert results[0].course == "Information Retrieval"
    assert results[0].location_type == "page"
    assert results[0].snippet == "distributed computation with MapReduce"
    assert results[0].vector_collection.startswith("document_index__")
    assert results[0].vector_id == f"chunk:{relevant.chunk_id}"
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score >= results[1].score

    safe = results[0].as_safe_dict()
    assert safe["snippet"] == "distributed computation with MapReduce"
    assert safe["vector_collection"] == results[0].vector_collection
    assert safe["vector_id"] == f"chunk:{relevant.chunk_id}"


def test_semantic_search_applies_course_index_and_top_k_filters(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        ir_doc = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="ir.md",
            source_type="document",
            text="distributed computation",
        )
        insert_minimal_chunk(
            connection,
            config,
            course_name="High Preformance Computing for Big Data",
            filename="hpc.pptx",
            extension=".pptx",
            category="slides",
            source_type="slides",
            text="distributed computation",
        )
        connection.commit()
    sync_vector_index(config)

    course_filtered = semantic_search(
        config, "distributed computation", course="information retrieval"
    )
    index_filtered = semantic_search(
        config, "distributed computation", indexes=["document_index"]
    )
    top_k_limited = semantic_search(config, "distributed computation", top_k=1)

    assert [r.chunk_id for r in course_filtered] == [ir_doc.chunk_id]
    assert [r.chunk_id for r in index_filtered] == [ir_doc.chunk_id]
    assert len(top_k_limited) == 1
    assert semantic_search(config, "distributed computation", indexes=[]) == []


def test_semantic_search_course_filter_is_applied_before_top_k(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        # This exceeds the old 4x overfetch window for top_k=1. Chroma must
        # apply the canonical course filter before semantic top-K selection.
        for number in range(334):
            insert_minimal_chunk(
                connection,
                config,
                course_name="Other Course",
                filename=f"other-{number}.md",
                text="distributed computation",
            )
        target = insert_minimal_chunk(
            connection,
            config,
            course_name="Target Course",
            filename="target.md",
            text="distributed systems overview",
        )
        connection.commit()
    sync_vector_index(config)

    results = semantic_search(
        config,
        "distributed computation",
        course="Target Course",
        top_k=1,
    )

    assert [result.chunk_id for result in results] == [target.chunk_id]


def test_semantic_search_without_collections_returns_empty(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        insert_minimal_chunk(connection, config, filename="a.md", text="distributed")
        connection.commit()

    # No vector index built yet -> no Chroma collections.
    assert semantic_search(config, "distributed") == []


def test_semantic_search_excludes_non_current_chunks(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection, config, filename="stale.md", text="distributed computation"
        )
        connection.commit()
    sync_vector_index(config)

    with closing(connect_sqlite(config)) as connection:
        connection.execute(
            "UPDATE files SET index_status = 'failed' WHERE id = ?", (stored.file_id,)
        )
        connection.commit()

    # The vector + embedding row still exist, but the SQLite re-join drops the
    # chunk because its source file is no longer indexed.
    assert semantic_search(config, "distributed computation") == []


def test_semantic_search_rejects_invalid_input(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config):
        pass

    with pytest.raises(SemanticSearchError, match="top_k"):
        semantic_search(config, "distributed", top_k=0)
    with pytest.raises(SemanticSearchError, match="must not be empty"):
        semantic_search(config, "   ")
    with pytest.raises(SemanticSearchError, match="Unknown logical index"):
        semantic_search(config, "distributed", indexes=["slides"])


def test_semantic_search_does_not_persist_search_tables(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        insert_minimal_chunk(connection, config, filename="a.md", text="distributed")
        connection.commit()
    sync_vector_index(config)

    with closing(connect_sqlite(config)) as connection:
        before_runs = connection.execute("SELECT COUNT(*) FROM search_runs").fetchone()[
            0
        ]
        before_results = connection.execute(
            "SELECT COUNT(*) FROM search_results"
        ).fetchone()[0]

    results = semantic_search(config, "distributed")

    with closing(connect_sqlite(config)) as connection:
        after_runs = connection.execute("SELECT COUNT(*) FROM search_runs").fetchone()[
            0
        ]
        after_results = connection.execute(
            "SELECT COUNT(*) FROM search_results"
        ).fetchone()[0]

    assert results
    assert (after_runs, after_results) == (before_runs, before_results)


def test_keyword_result_emits_null_vector_fields() -> None:
    keyword = RetrievalResult(
        chunk_id=1,
        file_id=1,
        course="Information Retrieval",
        file_path="notes.md",
        source_type="document",
        location_type=None,
        location_value=None,
        rank=1,
        score=1.0,
        snippet="BM25",
    )
    safe = keyword.as_safe_dict()

    assert keyword.retrieval_method == "keyword"
    assert safe["vector_collection"] is None
    assert safe["vector_id"] is None
    assert {"chunk_id", "file_id", "score", "snippet"}.issubset(safe)


# --------------------------------------------------------------------------- #
# CLI integration (subprocess)
# --------------------------------------------------------------------------- #


def test_vector_cli_indexes_and_searches(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    course_dir = courses_root / "Information Retrieval"
    course_dir.mkdir(parents=True)
    (course_dir / "syllabus.txt").write_text(
        "distributed computation with MapReduce and BM25",
        encoding="utf-8",
    )
    env = clean_subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(data_dir / "runs"),
            "UNI_RAG_EMBEDDING_MODEL": "BAAI/bge-m3",
            "PYTHONPATH": _subprocess_shim_pythonpath(),
        }
    )

    assert _run_cli(env, "inventory", "run").returncode == 0
    assert _run_cli(env, "extract", "run").returncode == 0
    index_result = _run_cli(env, "index", "vector")
    json_result = _run_cli(
        env, "search", "semantic", "distributed computation", "--json"
    )

    assert index_result.returncode == 0, index_result.stderr
    assert "Vector index completed" in index_result.stdout
    assert "vectors_indexed: 1" in index_result.stdout
    assert _sqlite_count(data_dir, "embeddings") == 1

    index_events = _run_log_events(data_dir, "index-vector")
    assert [event["event"] for event in index_events] == [
        "vector_index_started",
        "vector_index_completed",
    ]
    assert index_events[-1]["count"] == 1
    assert index_events[0]["model"] == "BAAI/bge-m3"
    assert index_events[-1]["model"] == "BAAI/bge-m3"

    with sqlite3.connect(data_dir / "uni_rag.sqlite") as connection:
        embedding = connection.execute(
            """
            SELECT embedding_model, embedding_dim, vector_collection
            FROM embeddings
            """
        ).fetchone()
    assert embedding[0] == "BAAI/bge-m3"
    assert embedding[1] == TEST_EMBEDDING_DIMENSIONS["BAAI/bge-m3"]
    assert "__baai-bge-m3__" in embedding[2]

    assert json_result.returncode == 0, json_result.stderr
    payload = json.loads(json_result.stdout)
    assert payload[0]["retrieval_method"] == "semantic"
    assert payload[0]["course"] == "Information Retrieval"
    assert payload[0]["vector_id"]
    assert payload[0]["vector_collection"].startswith("document_index__")
    assert payload[0]["snippet"]

    # Semantic search must not persist retrieval traces.
    assert _sqlite_count(data_dir, "search_runs") == 0
    assert _sqlite_count(data_dir, "search_results") == 0

    search_events = _run_log_events(data_dir, "search-semantic")
    assert search_events[-2]["event"] == "semantic_search_started"
    assert search_events[-1]["event"] == "semantic_search_completed"
    assert search_events[-1]["result_count"] == 1


# --------------------------------------------------------------------------- #
# Notebook
# --------------------------------------------------------------------------- #


def test_vector_index_eda_notebook_is_valid_and_read_only() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "vector_index_eda.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    source_text = "\n".join(cell.get("source", "") for cell in notebook.cells)
    cell_ids = {cell.get("id") for cell in notebook.cells}

    assert "uv run -m uni_rag_agent index vector" in source_text
    assert "import pandas as pd" in source_text
    assert "import matplotlib.pyplot as plt" in source_text
    assert "read-only" in source_text.lower()
    assert "query_only" in source_text
    assert "chromadb" in source_text.lower()
    assert {
        "open-read-only-sqlite",
        "load-vector-index-tables",
        "vector-index-coverage",
        "chroma-collection-metadata",
        "embedding-model-consistency",
        "plot-vector-source-coverage",
        "plot-missing-vector-rows",
        "plot-embedding-model-distribution",
    }.issubset(cell_ids)
    assert all(not cell.get("outputs") for cell in notebook.cells)
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def _run_cli(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _sqlite_count(data_dir: Path, table: str) -> int:
    with sqlite3.connect(data_dir / "uni_rag.sqlite") as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _run_log_events(data_dir: Path, slug: str) -> list[dict[str, object]]:
    log_files = sorted((data_dir / "runs").glob(f"*-{slug}.jsonl"))
    assert log_files
    return [
        json.loads(line)
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
    ]


def _subprocess_shim_pythonpath() -> str:
    shim = REPO_ROOT / "tests" / "subprocess_shim"
    existing = os.environ.get("PYTHONPATH")
    return os.pathsep.join(part for part in (str(shim), existing) if part)


def test_sync_updates_drifted_course_metadata_for_reassigned_files(
    tmp_path: Path,
    patch_huggingface_loader: None,
) -> None:
    """A course-less file vectorized before inventory assigns it a course must
    become reachable under that course after a normal incremental sync."""
    from tests.sqlite_helpers import TEST_TIMESTAMP

    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            course_name=None,
            filename="companies.md",
            text="practical training companies",
        )
        connection.commit()
    sync_vector_index(config)

    with closing(connect_sqlite(config)) as connection:
        course_id = connection.execute(
            """
            INSERT INTO courses (
                name, path, file_count, total_bytes, created_at, updated_at
            )
            VALUES (?, ?, 1, 1, ?, ?)
            """,
            (
                "General Resources",
                str(config.courses_root),
                TEST_TIMESTAMP,
                TEST_TIMESTAMP,
            ),
        ).lastrowid
        connection.execute(
            "UPDATE files SET course_id = ? WHERE id = ?",
            (course_id, stored.file_id),
        )
        connection.commit()

    # Stale Chroma metadata still carries course="", so the course-scoped
    # query cannot reach the chunk yet.
    assert (
        semantic_search(config, "practical training", course="General Resources") == []
    )

    repaired = sync_vector_index(config)

    assert repaired.vectors_indexed == 0
    assert any("drifted course/path" in item for item in repaired.diagnostics)
    assert [
        result.chunk_id
        for result in semantic_search(
            config, "practical training", course="General Resources"
        )
    ] == [stored.chunk_id]

    stable = sync_vector_index(config)
    assert not any("drifted course/path" in item for item in stable.diagnostics)
