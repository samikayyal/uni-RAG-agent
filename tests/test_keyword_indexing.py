from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import nbformat
import pytest

from uni_rag_agent.indexing import (
    KeywordSearchError,
    keyword_search,
    sync_keyword_index,
)
from uni_rag_agent.storage import connect_sqlite
from tests.sqlite_helpers import insert_minimal_chunk
from tests.support import clean_subprocess_env, initialized_connection, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_keyword_rebuild_indexes_only_current_eligible_chunks(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        indexed = insert_minimal_chunk(
            connection,
            config,
            filename="notes.md",
            source_type="document",
            text="BM25 keyword ranking",
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
            text="failed should not appear",
        )
        insert_minimal_chunk(
            connection,
            config,
            filename="pending.md",
            index_status="pending",
            source_type="document",
            text="pending should not appear",
        )
        insert_minimal_chunk(
            connection,
            config,
            filename="empty.md",
            source_type="document",
            text="   ",
        )
        connection.commit()

    result = sync_keyword_index(config)

    assert result.rebuild is True
    assert result.rows_removed == 0
    assert result.chunks_seen == 3
    assert result.rows_indexed == 2
    assert result.by_source_type == {"data_schema": 1, "document": 1}
    assert result.diagnostics == (
        "Indexed 2 FTS rows from 3 current eligible chunks; "
        "blank chunk text is skipped.",
    )
    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute(
            "SELECT chunk_id, source_type FROM chunk_fts ORDER BY chunk_id"
        ).fetchall()
    assert [(row["chunk_id"], row["source_type"]) for row in rows] == [
        (indexed.chunk_id, "document"),
        (data_schema.chunk_id, "data_schema"),
    ]


def test_keyword_search_matches_projection_fields(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        inserted = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="pathterm.md",
            relative_path="lectures/pathterm.md",
            title="Ranking Models",
            text="BM25 ranks documents",
            location_type="page",
            location_value="3",
        )
        insert_minimal_chunk(
            connection,
            config,
            course_name="High Preformance Computing for Big Data",
            filename="mapreduce.md",
            text="MapReduce scheduling",
        )
        connection.commit()
    sync_keyword_index(config)

    title_results = keyword_search(config, "ranking")
    course_results = keyword_search(config, "retrieval")
    path_results = keyword_search(config, "pathterm")

    assert title_results[0].chunk_id == inserted.chunk_id, "title should be searchable"
    assert course_results[0].chunk_id == inserted.chunk_id, (
        "course name should be searchable"
    )
    assert path_results[0].chunk_id == inserted.chunk_id, (
        "file path should be searchable"
    )


def test_keyword_search_applies_exact_course_filter_and_result_contract(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        inserted = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="bm25.md",
            text="BM25 ranks documents",
            location_type="page",
            location_value="3",
        )
        insert_minimal_chunk(
            connection,
            config,
            course_name="High Preformance Computing for Big Data",
            filename="bm25-mapreduce.md",
            text="BM25 MapReduce scheduling",
        )
        connection.commit()
    sync_keyword_index(config)

    filtered_results = keyword_search(config, "bm25", course="information retrieval")
    partial_course_results = keyword_search(config, "bm25", course="information")

    assert filtered_results[0].course == "Information Retrieval"
    assert filtered_results[0].file_id == inserted.file_id
    assert partial_course_results == []
    assert filtered_results[0].location_type == "page"
    assert filtered_results[0].location_value == "3"
    assert filtered_results[0].retrieval_method == "keyword"
    assert filtered_results[0].as_safe_dict()["chunk_id"] == inserted.chunk_id


def test_keyword_search_plain_text_or_and_index_filters(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        slides = insert_minimal_chunk(
            connection,
            config,
            filename="lecture.pptx",
            extension=".pptx",
            category="slides",
            source_type="slides",
            text="MapReduce shuffle phase",
        )
        code = insert_minimal_chunk(
            connection,
            config,
            filename="vectors.py",
            extension=".py",
            category="code",
            source_type="code",
            text="vector search implementation",
        )
        connection.commit()
    sync_keyword_index(config)

    results = keyword_search(config, "mapreduce vector")
    slides_only = keyword_search(config, "mapreduce vector", indexes=["slides_index"])

    assert {result.chunk_id for result in results} == {slides.chunk_id, code.chunk_id}
    assert [result.chunk_id for result in slides_only] == [slides.chunk_id]
    assert keyword_search(config, "mapreduce", indexes=[]) == []
    with pytest.raises(KeywordSearchError, match="Unknown logical index"):
        keyword_search(config, "mapreduce", indexes=["slides"])


def test_keyword_search_ranking_score_and_deterministic_ties(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        both_terms = insert_minimal_chunk(
            connection,
            config,
            filename="both.md",
            text="BM25 MapReduce",
        )
        one_term = insert_minimal_chunk(
            connection,
            config,
            filename="one.md",
            text="BM25",
        )
        first_tie = insert_minimal_chunk(
            connection,
            config,
            filename="tie-a.md",
            text="deterministic",
        )
        second_tie = insert_minimal_chunk(
            connection,
            config,
            filename="tie-b.md",
            text="deterministic",
        )
        connection.commit()
    sync_keyword_index(config)

    mixed_results = keyword_search(config, "bm25 mapreduce", top_k=2)
    tie_results = keyword_search(config, "deterministic", top_k=2)

    assert mixed_results[0].chunk_id == both_terms.chunk_id
    assert mixed_results[1].chunk_id == one_term.chunk_id
    assert [result.rank for result in mixed_results] == [1, 2]
    assert all(result.score >= 0 for result in mixed_results)
    assert [result.chunk_id for result in tie_results] == [
        first_tie.chunk_id,
        second_tie.chunk_id,
    ]


def test_keyword_search_excludes_stale_fts_rows_and_does_not_persist_results(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config) as connection:
        stored = insert_minimal_chunk(
            connection,
            config,
            filename="stale.md",
            text="BM25 stale row",
        )
        connection.commit()
    sync_keyword_index(config)

    with closing(connect_sqlite(config)) as connection:
        connection.execute(
            "UPDATE files SET index_status = 'failed' WHERE id = ?",
            (stored.file_id,),
        )
        before_runs = _table_count(connection, "search_runs")
        before_results = _table_count(connection, "search_results")
        connection.commit()

    results = keyword_search(config, "BM25")

    with closing(connect_sqlite(config)) as connection:
        after_runs = _table_count(connection, "search_runs")
        after_results = _table_count(connection, "search_results")

    assert results == []
    assert after_runs == before_runs
    assert after_results == before_results


def test_keyword_search_rejects_empty_queries_and_invalid_limits(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config):
        pass

    with pytest.raises(KeywordSearchError, match="at least one word or number"):
        keyword_search(config, "?! ")
    with pytest.raises(KeywordSearchError, match="top_k"):
        keyword_search(config, "bm25", top_k=0)


def test_empty_keyword_index_returns_no_results_without_crashing(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    with initialized_connection(config):
        pass

    result = sync_keyword_index(config)

    assert result.rows_indexed == 0
    assert result.diagnostics
    assert keyword_search(config, "bm25") == []


def test_keyword_cli_indexes_and_searches(
    tmp_path: Path,
) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    course_dir = courses_root / "Information Retrieval"
    course_dir.mkdir(parents=True)
    (course_dir / "syllabus.txt").write_text(
        "BM25 keyword search and MapReduce",
        encoding="utf-8",
    )
    env = clean_subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(data_dir / "runs"),
            # Override any repo-root .env model selection so this test keeps
            # exercising the intentionally unconfigured semantic CLI path.
            "UNI_RAG_EMBEDDING_MODEL": "",
        }
    )

    inventory_result = _run_cli(env, "inventory", "run")
    extract_result = _run_cli(env, "extract", "run")
    index_result = _run_cli(env, "index", "keyword")
    rebuild_result = _run_cli(env, "index", "keyword", "--rebuild")
    table_result = _run_cli(env, "search", "keyword", "BM25")
    json_result = _run_cli(env, "search", "keyword", "BM25", "--json")
    semantic_result = _run_cli(env, "search", "semantic", "BM25")
    semantic_json_result = _run_cli(env, "search", "semantic", "BM25", "--json")

    assert inventory_result.returncode == 0, inventory_result.stderr
    assert extract_result.returncode == 0, extract_result.stderr
    assert index_result.returncode == 0, index_result.stderr
    assert "Keyword index completed" in index_result.stdout
    assert "rows_indexed: 1" in index_result.stdout
    assert rebuild_result.returncode == 0, rebuild_result.stderr
    assert _sqlite_count(data_dir, "chunk_fts") == 1

    events = _run_log_events(data_dir, "index-keyword")
    assert [event["event"] for event in events] == [
        "keyword_index_started",
        "keyword_index_completed",
    ]
    assert events[-1]["rows_indexed"] == 1
    assert events[-1]["count"] == 1

    assert table_result.returncode == 0, table_result.stderr
    assert "rank | score | chunk_id" in table_result.stdout
    assert "BM25" in table_result.stdout

    assert json_result.returncode == 0, json_result.stderr
    payload = json.loads(json_result.stdout)
    assert payload[0]["retrieval_method"] == "keyword"
    assert payload[0]["course"] == "Information Retrieval"
    assert {"chunk_id", "file_id", "score", "snippet"}.issubset(payload[0])

    search_events = _run_log_events(data_dir, "search-keyword")
    assert search_events[-2]["event"] == "keyword_search_started"
    assert search_events[-1]["event"] == "keyword_search_completed"
    assert search_events[-1]["keyword_terms"] == ["BM25"]
    assert search_events[-1]["count"] == 1
    assert search_events[-1]["result_count"] == 1

    # Semantic search now requires an explicitly selected or configured model,
    # even when no vector collections have been created yet.
    assert semantic_result.returncode == 7
    assert "No embedding model selected" in semantic_result.stderr
    assert semantic_json_result.returncode == 7
    assert "No embedding model selected" in semantic_json_result.stderr
    semantic_events = _run_log_events(data_dir, "search-semantic")
    assert semantic_events[0]["model"] == "(unset)"

    # Keyword JSON now carries the shared nullable vector fields without
    # breaking existing subset assertions.
    assert payload[0]["vector_collection"] is None
    assert payload[0]["vector_id"] is None


def test_keyword_index_eda_notebook_is_valid_and_read_only() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "keyword_index_eda.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    source_text = "\n".join(cell.get("source", "") for cell in notebook.cells)
    cell_ids = {cell.get("id") for cell in notebook.cells}

    assert "uv run -m uni_rag_agent index keyword" in source_text
    assert "import pandas as pd" in source_text
    assert "import matplotlib.pyplot as plt" in source_text
    assert "read-only" in source_text.lower()
    assert "query_only" in source_text
    assert {
        "open-read-only-sqlite",
        "load-keyword-index-tables",
        "keyword-index-coverage",
        "keyword-query-smoke-results",
        "plot-keyword-source-coverage",
        "plot-missing-keyword-rows",
        "plot-keyword-smoke-results",
    }.issubset(cell_ids)
    assert all(not cell.get("outputs") for cell in notebook.cells)
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _sqlite_count(data_dir: Path, table: str) -> int:
    with sqlite3.connect(data_dir / "uni_rag.sqlite") as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _run_cli(
    env: dict[str, str],
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_log_events(data_dir: Path, slug: str) -> list[dict[str, object]]:
    log_files = sorted((data_dir / "runs").glob(f"*-{slug}.jsonl"))
    assert log_files
    return [
        json.loads(line)
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
    ]
