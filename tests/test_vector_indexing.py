from __future__ import annotations

import dataclasses
import json
import math
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import nbformat
import pytest

from uni_rag_agent.config import Config, load_config
from uni_rag_agent.indexing import (
    REAL_EMBEDDING_PROFILES,
    FakeDeterministicEmbeddings,
    SemanticSearchError,
    VectorIndexError,
    build_embedding_model,
    get_embedding_model,
    physical_collection_name,
    resolve_embedding_profile,
    semantic_search,
    sync_vector_index,
)
from uni_rag_agent.indexing import embeddings as embeddings_module
from uni_rag_agent.retrieval import RetrievalResult
from uni_rag_agent.storage import connect_sqlite, ensure_data_dirs, initialize_schema
from tests.sqlite_helpers import insert_minimal_chunk

REPO_ROOT = Path(__file__).resolve().parents[1]
UNI_RAG_ENV_PREFIX = "UNI_RAG_"


def make_config(tmp_path: Path, **overrides: object) -> Config:
    courses = tmp_path / "Courses"
    courses.mkdir(exist_ok=True)
    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    if overrides:
        config = dataclasses.replace(config, **overrides)
    return config


def _initialized_connection(config: Config) -> sqlite3.Connection:
    ensure_data_dirs(config)
    connection = connect_sqlite(config)
    initialize_schema(connection)
    return connection


# --------------------------------------------------------------------------- #
# Embedding profile registry (pure, offline)
# --------------------------------------------------------------------------- #


def test_resolve_profile_follows_config_fake_default(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    profile = resolve_embedding_profile(config, None)

    assert profile.is_fake is True
    assert profile.provider == "fake"
    assert profile.dimension == config.embedding_dim
    assert profile.metric == "cosine"


def test_explicit_fake_model_selects_fake_profile(tmp_path: Path) -> None:
    config = make_config(tmp_path, use_fake_embeddings=False, embedding_model="x")
    profile = resolve_embedding_profile(config, "fake-embedding")

    assert profile.is_fake is True


def test_explicit_real_model_overrides_fake_default(tmp_path: Path) -> None:
    config = make_config(tmp_path)  # use_fake_embeddings defaults to True
    profile = resolve_embedding_profile(config, "BAAI/bge-m3")

    assert profile.is_fake is False
    assert profile.provider == "huggingface"
    assert profile.requires_extra == "embeddings"
    assert profile.dimension == 1024


def test_unknown_model_fails_clearly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(VectorIndexError, match="Unknown embedding model"):
        resolve_embedding_profile(config, "no/such-model")


def test_fake_disabled_with_unknown_config_model_fails_clearly(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        use_fake_embeddings=False,
        embedding_provider="huggingface",
        embedding_model="fake-embedding",
    )
    with pytest.raises(SemanticSearchError, match="Fake embeddings are disabled"):
        resolve_embedding_profile(config, None, error=SemanticSearchError)


def test_real_profile_registry_metadata() -> None:
    assert set(REAL_EMBEDDING_PROFILES) == {
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3",
        "jinaai/jina-embeddings-v5-text-small",
        "google/embeddinggemma-300m",
    }
    for profile in REAL_EMBEDDING_PROFILES.values():
        assert profile.provider == "huggingface"
        assert profile.is_fake is False
        assert profile.requires_extra == "embeddings"
        assert profile.metric == "cosine"
        assert profile.access_notes

    assert REAL_EMBEDDING_PROFILES["BAAI/bge-m3"].dimension == 1024
    assert (
        REAL_EMBEDDING_PROFILES["jinaai/jina-embeddings-v3"].trust_remote_code is True
    )
    gemma = REAL_EMBEDDING_PROFILES["google/embeddinggemma-300m"]
    assert gemma.gated is True
    assert gemma.dimension == 768


# --------------------------------------------------------------------------- #
# Fake adapter (deterministic, offline)
# --------------------------------------------------------------------------- #


def test_fake_embeddings_are_deterministic_and_normalized(tmp_path: Path) -> None:
    fake = FakeDeterministicEmbeddings(64)
    first = fake.embed_query("distributed computation")
    second = fake.embed_query("distributed computation")

    assert first == second
    assert len(first) == 64
    assert math.isclose(
        math.sqrt(sum(value * value for value in first)), 1.0, rel_tol=1e-9
    )

    # Identical text -> cosine distance 0 (dot product 1).
    assert math.isclose(sum(a * b for a, b in zip(first, second)), 1.0, rel_tol=1e-9)
    # Disjoint tokens -> much lower similarity than an exact match.
    disjoint = fake.embed_query("xylophone qwerty")
    assert sum(a * b for a, b in zip(first, disjoint)) < 0.5


def test_get_embedding_model_uses_config_dimension(tmp_path: Path) -> None:
    config = make_config(tmp_path, embedding_dim=128)
    built = build_embedding_model(config)

    assert isinstance(built.embeddings, FakeDeterministicEmbeddings)
    assert built.dimension == 128
    assert len(get_embedding_model(config).embed_query("probe")) == 128


# --------------------------------------------------------------------------- #
# Optional-dependency failure path (monkeypatched, never loads a real model)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def force_missing_embeddings_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require(profile: object, *, error: type[Exception]) -> object:
        raise error(
            "Embedding model requires the optional 'embeddings' extra. "
            "Install it with: uv sync --extra embeddings"
        )

    monkeypatch.setattr(embeddings_module, "_require_huggingface", fake_require)


def test_real_model_missing_extra_fails_clearly_for_get_model(
    tmp_path: Path,
    force_missing_embeddings_extra: None,
) -> None:
    config = make_config(tmp_path)
    with pytest.raises(VectorIndexError, match="uv sync --extra embeddings"):
        get_embedding_model(config, "BAAI/bge-m3")


def test_real_model_missing_extra_fails_clearly_for_sync_and_search(
    tmp_path: Path,
    force_missing_embeddings_extra: None,
) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config):
        pass

    with pytest.raises(VectorIndexError, match="embeddings' extra"):
        sync_vector_index(config, model="BAAI/bge-m3")
    with pytest.raises(SemanticSearchError, match="embeddings' extra"):
        semantic_search(config, "distributed", model="BAAI/bge-m3")


# --------------------------------------------------------------------------- #
# Physical collection naming
# --------------------------------------------------------------------------- #


def test_physical_collection_name_is_model_namespaced_and_stable() -> None:
    fake = physical_collection_name(
        "document_index",
        provider="fake",
        model_name="fake-embedding",
        dimension=384,
        metric="cosine",
    )
    same = physical_collection_name(
        "document_index",
        provider="fake",
        model_name="fake-embedding",
        dimension=384,
        metric="cosine",
    )
    real = physical_collection_name(
        "document_index",
        provider="huggingface",
        model_name="BAAI/bge-m3",
        dimension=1024,
        metric="cosine",
    )

    assert fake == same
    assert fake != real
    assert fake.startswith("document_index__fake-embedding__")
    assert real.startswith("document_index__baai-bge-m3__")


# --------------------------------------------------------------------------- #
# Vector sync (fake embeddings + ChromaDB)
# --------------------------------------------------------------------------- #


def test_sync_indexes_only_current_eligible_chunks(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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
    assert result.model == "fake-embedding"
    assert result.provider == "fake"
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
        assert row["embedding_model"] == "fake-embedding"
        assert row["embedding_dim"] == config.embedding_dim
    document_row = next(r for r in rows if r["chunk_id"] == document.chunk_id)
    assert document_row["vector_collection"].startswith("document_index__")


def test_sync_is_idempotent(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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


def test_sync_collection_filter_limits_to_one_logical_index(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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


def test_sync_unknown_collection_fails_clearly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config):
        pass
    with pytest.raises(VectorIndexError, match="Unknown logical index"):
        sync_vector_index(config, collection="slides")


def test_rebuild_removes_stale_rows_and_repopulates_selected_model(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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


def test_sync_reports_no_eligible_chunks(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config):
        pass

    result = sync_vector_index(config)

    assert result.vectors_indexed == 0
    assert result.chunks_seen == 0
    assert any("No eligible indexed chunks" in d for d in result.diagnostics)


# --------------------------------------------------------------------------- #
# Semantic search (fake embeddings)
# --------------------------------------------------------------------------- #


def test_semantic_search_returns_sqlite_joined_results(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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


def test_semantic_search_applies_course_index_and_top_k_filters(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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


def test_semantic_search_without_collections_returns_empty(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
        insert_minimal_chunk(connection, config, filename="a.md", text="distributed")
        connection.commit()

    # No vector index built yet -> no Chroma collections.
    assert semantic_search(config, "distributed") == []


def test_semantic_search_excludes_non_current_chunks(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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


def test_semantic_search_rejects_invalid_input(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config):
        pass

    with pytest.raises(SemanticSearchError, match="top_k"):
        semantic_search(config, "distributed", top_k=0)
    with pytest.raises(SemanticSearchError, match="must not be empty"):
        semantic_search(config, "   ")
    with pytest.raises(SemanticSearchError, match="Unknown logical index"):
        semantic_search(config, "distributed", indexes=["slides"])


def test_semantic_search_does_not_persist_search_tables(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with _initialized_connection(config) as connection:
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
    env = _subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(data_dir / "runs"),
            "UNI_RAG_USE_FAKE_LLM": "true",
            "UNI_RAG_USE_FAKE_EMBEDDINGS": "true",
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
    assert index_events[-1]["model"] == "fake-embedding"

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


def _subprocess_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(UNI_RAG_ENV_PREFIX)
    }
    env.update(overrides)
    return env
