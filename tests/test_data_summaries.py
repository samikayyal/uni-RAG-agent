from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import nbformat

from uni_rag_agent.config import Config, load_config
from uni_rag_agent.extraction import (
    summarize_data_files,
    summarize_jsonl,
    summarize_sqlite,
)
from uni_rag_agent.inventory import inventory_courses
from uni_rag_agent.storage import connect_sqlite

REPO_ROOT = Path(__file__).resolve().parents[1]
UNI_RAG_ENV_PREFIX = "UNI_RAG_"


def make_config(tmp_path: Path) -> Config:
    (tmp_path / "Courses").mkdir()
    return load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")


def test_data_summary_run_processes_supported_formats_and_chunks(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()

    _write_csv(course_dir / "ranking.csv")
    _write_xlsx(course_dir / "workbook.xlsx")
    _write_json_array(course_dir / "records.json")
    (course_dir / "settings.json").write_text(
        json.dumps({"course": "IR", "threshold": 0.7, "enabled": True}),
        encoding="utf-8",
    )
    _write_jsonl(course_dir / "events.jsonl")
    _write_sqlite(course_dir / "warehouse.sqlite")

    inventory_courses(config)
    result = summarize_data_files(config)

    assert result.status == "completed", "data-summary run status"
    assert result.files_seen == 6, "data-summary files seen"
    assert result.files_indexed == 6, "data-summary files indexed"
    assert result.files_failed == 0, "data-summary files failed"
    assert result.summaries_created == 6, "data summaries created"
    assert result.chunks_created == 8, "data-summary chunks created"
    assert result.by_format == {
        "csv": 1,
        "json": 2,
        "jsonl": 1,
        "sqlite": 1,
        "xlsx": 1,
    }, "data-summary counts by format"

    with closing(connect_sqlite(config)) as connection:
        summary_rows = connection.execute(
            """
            SELECT files.filename, files.index_status, data_summaries.*
            FROM data_summaries
            JOIN files ON files.id = data_summaries.file_id
            ORDER BY files.filename
            """
        ).fetchall()
        chunk_rows = connection.execute(
            """
            SELECT files.filename, chunks.source_type, chunks.location_type,
                   chunks.location_value, chunks.title, chunks.text
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            ORDER BY files.filename, chunks.chunk_index
            """
        ).fetchall()
        run_row = connection.execute(
            "SELECT config_json, files_seen, files_indexed FROM extraction_runs WHERE id = ?",
            (result.run_id,),
        ).fetchone()

    summaries_by_file = {row["filename"]: row for row in summary_rows}
    assert set(summaries_by_file) == {
        "events.jsonl",
        "ranking.csv",
        "records.json",
        "settings.json",
        "warehouse.sqlite",
        "workbook.xlsx",
    }, "summary rows by filename"
    assert all(row["index_status"] == "indexed" for row in summary_rows), (
        "data-schema files indexed"
    )

    csv_summary = summaries_by_file["ranking.csv"]
    csv_schema = json.loads(csv_summary["schema_json"])
    assert csv_summary["row_count"] == 6, "csv row count"
    assert csv_summary["column_count"] == 2, "csv column count"
    assert csv_schema["sections"][0]["columns"][0]["name"] == "term", "csv column"
    assert csv_schema["sections"][0]["columns"][1]["type"] == "integer", "csv type"
    assert "bm25" in csv_summary["summary_text"], "csv sample row included"
    assert "late_only" not in csv_summary["summary_text"], "csv sample row limit"

    xlsx_summary = summaries_by_file["workbook.xlsx"]
    assert xlsx_summary["sheet_count"] == 2, "xlsx sheet count"
    assert xlsx_summary["row_count"] == 8, "xlsx row count"

    sqlite_summary = summaries_by_file["warehouse.sqlite"]
    sqlite_schema = json.loads(sqlite_summary["schema_json"])
    assert sqlite_summary["table_count"] == 2, "sqlite table count"
    assert {section["name"] for section in sqlite_schema["sections"]} == {
        "metrics",
        "users",
    }, "sqlite table names"

    chunks_by_file: dict[str, list] = {}
    for row in chunk_rows:
        chunks_by_file.setdefault(row["filename"], []).append(row)
        assert row["source_type"] == "data_schema", f"{row['filename']} source type"
        assert row["location_type"] == "schema", f"{row['filename']} location type"

    assert [row["location_value"] for row in chunks_by_file["workbook.xlsx"]] == [
        "Grades",
        "Topics",
    ], "xlsx schema chunk locations"
    assert [row["location_value"] for row in chunks_by_file["warehouse.sqlite"]] == [
        "metrics",
        "users",
    ], "sqlite schema chunk locations"
    assert "late_only" not in "\n".join(row["text"] for row in chunk_rows), (
        "chunk sample row limit"
    )
    assert "Metric" in chunks_by_file["warehouse.sqlite"][0]["text"], "sqlite sample"

    run_payload = json.loads(run_row["config_json"])
    assert run_payload["run_type"] == "data_summary", "data-summary run payload"
    assert run_payload["sample_row_limit"] == 5, "data-summary sample row limit"
    assert run_row["files_seen"] == 6, "stored data-summary files seen"
    assert run_row["files_indexed"] == 6, "stored data-summary files indexed"


def test_jsonl_summary_merges_heterogeneous_sample_schema(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        {"event": "keyword", "rank": 1},
        {"event": "semantic", "score": 0.82},
        {"metadata": {"course": "IR"}, "tags": ["search", "ranking"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = summarize_jsonl(path, file_id=7)
    schema = json.loads(summary.schema_json)
    columns = {
        column["name"]: column["type"] for column in schema["sections"][0]["columns"]
    }

    assert summary.row_count == 3, "jsonl heterogeneous row count"
    assert columns == {
        "event": "string",
        "rank": "integer",
        "score": "number",
        "metadata": "object",
        "tags": "array",
    }, "jsonl heterogeneous column schema"
    assert "metadata (object)" in summary.summary_text, "jsonl object column text"
    assert "tags (array)" in summary.summary_text, "jsonl array column text"


def test_sqlite_summary_handles_database_with_no_user_tables(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite"
    with sqlite3.connect(path):
        pass

    summary = summarize_sqlite(path, file_id=8)
    schema = json.loads(summary.schema_json)

    assert summary.table_count == 0, "empty sqlite table count"
    assert summary.row_count == 0, "empty sqlite row count"
    assert summary.column_count == 0, "empty sqlite column count"
    assert schema["sections"] == [], "empty sqlite sections"
    assert "Tables: 0" in summary.summary_text, "empty sqlite summary text"


def test_data_summary_failure_does_not_abort_run(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    _write_csv(course_dir / "ranking.csv")
    (course_dir / "broken.json").write_text("{not json", encoding="utf-8")

    inventory_courses(config)
    result = summarize_data_files(config)

    assert result.status == "completed"
    assert result.files_seen == 2
    assert result.files_indexed == 1
    assert result.files_failed == 1
    assert "invalid JSON file" in result.failures[0].error

    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute(
            """
            SELECT filename, index_status, reason_not_indexed
            FROM files
            ORDER BY filename
            """
        ).fetchall()

    by_name = {row["filename"]: row for row in rows}
    assert by_name["ranking.csv"]["index_status"] == "indexed"
    assert by_name["broken.json"]["index_status"] == "failed"
    assert "invalid JSON file" in by_name["broken.json"]["reason_not_indexed"]


def test_data_summary_file_id_filter_reports_nonpending_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    _write_csv(course_dir / "ranking.csv")

    inventory_courses(config)
    first = summarize_data_files(config)
    file_id = _file_id_for(config, "ranking.csv")
    second = summarize_data_files(config, file_id=file_id)

    assert first.files_indexed == 1
    assert second.files_seen == 0
    assert second.files_indexed == 0
    assert second.diagnostics
    assert "not pending" in second.diagnostics[0]


def test_data_summary_cli_writes_run_log(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    course_dir = courses_root / "Information Retrieval"
    course_dir.mkdir(parents=True)
    _write_csv(course_dir / "ranking.csv")
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

    inventory_result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "inventory", "run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    summary_result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "extract", "data-summaries"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert inventory_result.returncode == 0, inventory_result.stderr
    assert summary_result.returncode == 0, summary_result.stderr
    assert summary_result.stdout.strip()
    events = _run_log_events(data_dir, "extract-data-summaries")
    assert [event["event"] for event in events] == [
        "data_summary_started",
        "data_summary_completed",
    ]
    assert events[-1]["count"] == 1
    with sqlite3.connect(data_dir / "uni_rag.sqlite") as connection:
        summary_count = connection.execute(
            "SELECT COUNT(*) FROM data_summaries"
        ).fetchone()
        chunk_count = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_type = 'data_schema'"
        ).fetchone()
    assert summary_count[0] == 1
    assert chunk_count[0] == 1


def test_data_schema_eda_notebook_is_valid_and_read_only() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "data_schema_eda.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    source_text = "\n".join(cell.get("source", "") for cell in notebook.cells)
    cell_ids = {cell.get("id") for cell in notebook.cells}

    assert "import pandas as pd" in source_text
    assert "read-only" in source_text.lower()
    assert "query_only" in source_text
    assert {
        "load-data-summary-tables",
        "plot-summary-coverage",
        "plot-row-column-counts",
        "plot-sample-coverage",
        "plot-failed-data-files",
    }.issubset(cell_ids)
    assert all(not cell.get("outputs") for cell in notebook.cells)
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def _write_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "term,score",
                "bm25,1",
                "vector,2",
                "rrf,3",
                "mapreduce,4",
                "hdfs,5",
                "late_only,6",
            ]
        ),
        encoding="utf-8",
    )


def _write_xlsx(path: Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    grades = workbook.active
    grades.title = "Grades"
    grades.append(["student", "score"])
    for index in range(1, 7):
        grades.append([f"s{index}", index])
    topics = workbook.create_sheet("Topics")
    topics.append(["topic", "week"])
    topics.append(["BM25", 1])
    topics.append(["RRF", 2])
    workbook.save(path)


def _write_json_array(path: Path) -> None:
    payload = [
        {"term": "bm25", "score": 1},
        {"term": "vector", "score": 2},
        {"term": "rrf", "score": 3},
        {"term": "mapreduce", "score": 4},
        {"term": "hdfs", "score": 5},
        {"term": "late_only", "score": 6},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path) -> None:
    rows = [
        {"event": "bm25", "rank": 1},
        {"event": "vector", "rank": 2},
        {"event": "rrf", "rank": 3},
        {"event": "mapreduce", "rank": 4},
        {"event": "hdfs", "rank": 5},
        {"event": "late_only", "rank": 6},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )


def _write_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metrics (name TEXT, value INTEGER)")
        connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany(
            "INSERT INTO metrics (name, value) VALUES (?, ?)",
            [
                ("Metric", 1),
                ("Recall", 2),
                ("Precision", 3),
                ("F1", 4),
                ("NDCG", 5),
                ("late_only", 6),
            ],
        )
        connection.executemany(
            "INSERT INTO users (name) VALUES (?)",
            [("Ada",), ("Grace",)],
        )


def _file_id_for(config: Config, filename: str) -> int:
    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            "SELECT id FROM files WHERE filename = ?",
            (filename,),
        ).fetchone()
    return int(row["id"])


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
