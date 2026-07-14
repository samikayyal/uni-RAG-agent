from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from collections.abc import Mapping
from pathlib import Path

import pytest
import uni_rag_agent
import uni_rag_agent.cli as cli
from tests.support import clean_subprocess_env
from tests.support import make_config
from uni_rag_agent.answering import AnswerGenerationError, AnswerResult
from uni_rag_agent.retrieval import EvidenceError
from uni_rag_agent.storage import StorageError

REPO_ROOT = Path(__file__).resolve().parents[1]


class _CaptureLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def info(self, message: str, *, extra: dict[str, object]) -> None:
        del message
        self.events.append(extra)

    def error(self, message: str, *, extra: dict[str, object]) -> None:
        del message
        self.events.append(extra)

    def exception(self, message: str, *, extra: dict[str, object]) -> None:
        raise AssertionError(f"unsafe exception telemetry used for {message}: {extra}")


def run_cli(
    *args: str,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", *args],
        cwd=REPO_ROOT,
        env=clean_subprocess_env(env),
        text=True,
        capture_output=True,
        check=False,
    )


def test_package_importable() -> None:
    assert uni_rag_agent.__version__


def test_help_exits_successfully() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "uv run -m uni_rag_agent config check" in result.stdout
    assert "uv run -m uni_rag_agent inventory run" in result.stdout
    assert "evidence" in result.stdout


@pytest.mark.parametrize(
    ("args", "expected"),
    (
        (("evidence", "build", "--help"), "--json"),
        (("evidence", "show", "--help"), "--search-run-id"),
    ),
)
def test_evidence_cli_help_exposes_contract_flags(
    args: tuple[str, ...],
    expected: str,
) -> None:
    result = run_cli(*args)

    assert result.returncode == 0
    assert expected in result.stdout


def test_evidence_build_handler_emits_one_safe_json_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )

    class CaptureLogger:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def info(self, message: str, *, extra: dict[str, object]) -> None:
            del message
            self.events.append(extra)

        def exception(self, message: str, *, extra: dict[str, object]) -> None:
            del message
            self.events.append(extra)

    logger = CaptureLogger()
    result = SimpleNamespace(
        search_run_id=11,
        evidence_packet_id=22,
        coverage=SimpleNamespace(
            status="completed",
            evidence_count=0,
            fused_candidate_count=0,
            token_budget_omission_count=0,
            oversized_evidence_omission_count=0,
        ),
        retrieval_run=SimpleNamespace(result_sets=()),
        as_safe_dict=lambda: {"search_run_id": 11, "evidence_packet_id": 22},
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_command_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(cli, "build_evidence", lambda *args, **kwargs: result)

    return_code = cli.main(
        ["evidence", "build", "safe query", "--model", "BAAI/bge-m3", "--json"]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "search_run_id": 11,
        "evidence_packet_id": 22,
    }
    assert logger.events[-1]["event"] == "evidence_build_completed"


def test_evidence_show_failure_uses_packet_load_error_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )

    class CaptureLogger:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def info(self, message: str, *, extra: dict[str, object]) -> None:
            del message
            self.events.append(extra)

        def exception(self, message: str, *, extra: dict[str, object]) -> None:
            del message
            self.events.append(extra)

    logger = CaptureLogger()
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_command_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(
        cli,
        "load_evidence_packet",
        lambda *args, **kwargs: (_ for _ in ()).throw(EvidenceError("missing packet")),
    )

    return_code = cli.main(["evidence", "show", "--search-run-id", "11"])

    captured = capsys.readouterr()
    assert return_code == cli.EVIDENCE_ERROR
    assert "Evidence error: missing packet" in captured.err
    assert logger.events[-1]["event"] == "evidence_packet_load_failed"


def test_answer_handler_emits_one_json_result_with_audit_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(tmp_path)
    logger = _CaptureLogger()
    packet = SimpleNamespace(search_run_id=11)
    answer = AnswerResult(
        answer_text="Insufficient evidence.",
        limitations=("No evidence.",),
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_command_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(cli, "load_evidence_packet", lambda *args, **kwargs: packet)
    monkeypatch.setattr(cli, "generate_answer", lambda *args, **kwargs: answer)
    monkeypatch.setattr(cli, "store_answer", lambda *args, **kwargs: 33)

    return_code = cli.main(["answer", "--evidence-packet-id", "22", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert return_code == cli.SUCCESS
    assert captured.err == ""
    assert payload["answer_id"] == 33
    assert payload["evidence_packet_id"] == 22
    assert payload["search_run_id"] == 11
    assert logger.events[-1]["event"] == "answer_completed"


def test_ask_answer_failure_returns_answer_error_and_keeps_packet_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="planner",
        answer_llm_provider="ollama",
        answer_llm_model="answerer",
    )
    logger = _CaptureLogger()
    evidence_result = SimpleNamespace(
        packet=object(),
        evidence_packet_id=22,
        search_run_id=11,
    )
    store_called = False

    def fail_store(*args, **kwargs):
        nonlocal store_called
        store_called = True

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_command_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(cli, "build_evidence", lambda *args, **kwargs: evidence_result)
    monkeypatch.setattr(
        cli,
        "generate_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AnswerGenerationError("provider down")
        ),
    )
    monkeypatch.setattr(cli, "store_answer", fail_store)

    return_code = cli.main(["ask", "grounded query", "--model", "BAAI/bge-m3"])

    captured = capsys.readouterr()
    assert return_code == cli.ANSWER_ERROR
    assert "Answer error: provider down" in captured.err
    assert not store_called
    assert logger.events[-1]["event"] == "ask_answer_failed"
    assert logger.events[-1]["evidence_packet_id"] == 22


def test_ask_storage_failure_uses_sanitized_error_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    logger = _CaptureLogger()
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_command_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(
        cli,
        "build_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            StorageError("D:\\private\\database.sqlite failed")
        ),
    )

    return_code = cli.main(["ask", "query", "--model", "BAAI/bge-m3"])

    captured = capsys.readouterr()
    assert return_code == cli.STORAGE_ERROR
    assert "Storage error:" in captured.err
    assert logger.events[-1]["event"] == "ask_failed"
    assert logger.events[-1]["answer_error"] == "StorageError"
    assert "exception" not in logger.events[-1]


def test_retrieval_eda_notebook_contract_is_read_only_and_output_free() -> None:
    notebook = json.loads(
        (REPO_ROOT / "notebooks" / "retrieval_eda.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells
    )
    assert "mode=ro" in source
    assert "PRAGMA query_only = ON" in source
    assert "search_result_sets" in source
    assert not any(
        keyword in source.upper()
        for keyword in (
            "INSERT INTO",
            "UPDATE ",
            "DELETE FROM",
            "CREATE TABLE",
            "DROP TABLE",
        )
    )


def test_unknown_command_returns_nonzero_with_message() -> None:
    result = run_cli("unknown")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "unknown" in result.stderr


def test_retrieve_requires_an_explicit_embedding_model() -> None:
    result = run_cli("retrieve", "query text")

    assert result.returncode == 7
    assert "No embedding model selected" in result.stderr


def test_run_cli_ignores_unrelated_host_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNI_RAG_EMBEDDING_MODEL", "BAAI/bge-m3")

    result = run_cli("retrieve", "query text")

    assert result.returncode == 7
    assert "No embedding model selected" in result.stderr


def test_inventory_run_cli_fills_temp_storage(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    course_dir = courses_root / "Information Retrieval"
    course_dir.mkdir(parents=True)
    (course_dir / "syllabus.txt").write_text("BM25", encoding="utf-8")
    (course_dir / "lecture.mp4").write_bytes(b"mp4")
    env = clean_subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(data_dir / "runs"),
        }
    )

    run_result = run_cli("inventory", "run", env=env)
    summary_result = run_cli("inventory", "summary", env=env)

    assert run_result.returncode == 0, run_result.stderr
    assert run_result.stdout.strip()
    assert (data_dir / "uni_rag.sqlite").is_file()
    assert _sqlite_count(data_dir, "courses") == 1
    assert _sqlite_count(data_dir, "files") == 2
    assert _sqlite_count(data_dir, "files", "index_status = 'pending'") == 1
    assert _sqlite_count(data_dir, "files", "index_status = 'metadata_only'") == 1
    inventory_events = _run_log_events(data_dir, "inventory-run")
    assert [event["event"] for event in inventory_events] == [
        "inventory_started",
        "inventory_completed",
    ]
    assert inventory_events[-1]["run_id"] == 1
    assert inventory_events[-1]["count"] == 2

    assert summary_result.returncode == 0, summary_result.stderr
    assert summary_result.stdout.strip()


def test_inventory_cli_validates_paths_before_creating_run_log(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    courses_root.mkdir()
    env = clean_subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(courses_root / "runs"),
        }
    )

    result = run_cli("inventory", "run", env=env)

    assert result.returncode == 2
    assert "must not point inside the Courses root" in result.stderr
    assert not (courses_root / "runs").exists()


def test_extract_run_cli_writes_chunks_and_status(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    course_dir = courses_root / "Information Retrieval"
    course_dir.mkdir(parents=True)
    (course_dir / "syllabus.txt").write_text("BM25 keyword search", encoding="utf-8")
    env = clean_subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(data_dir / "runs"),
        }
    )

    inventory_result = run_cli("inventory", "run", env=env)
    extract_result = run_cli("extract", "run", env=env)
    status_result = run_cli("extract", "status", env=env)

    assert inventory_result.returncode == 0, inventory_result.stderr
    assert extract_result.returncode == 0, extract_result.stderr
    assert extract_result.stdout.strip()
    assert _sqlite_count(data_dir, "files", "index_status = 'indexed'") == 1
    assert _sqlite_count(data_dir, "chunks") == 1
    extraction_events = _run_log_events(data_dir, "extract-run")
    assert [event["event"] for event in extraction_events] == [
        "extraction_started",
        "extraction_completed",
    ]
    assert extraction_events[-1]["count"] == 1

    assert status_result.returncode == 0, status_result.stderr
    assert status_result.stdout.strip()


def test_env_example_exists_and_env_is_ignored() -> None:
    assert (REPO_ROOT / ".env.example").is_file()

    ignored_patterns = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert ".env" in ignored_patterns
    assert ".env.example" not in ignored_patterns


def _sqlite_count(data_dir: Path, table: str, where: str | None = None) -> int:
    query = f"SELECT COUNT(*) FROM {table}"
    if where is not None:
        query += f" WHERE {where}"
    with sqlite3.connect(data_dir / "uni_rag.sqlite") as connection:
        row = connection.execute(query).fetchone()
    return int(row[0])


def _run_log_events(data_dir: Path, slug: str) -> list[dict[str, object]]:
    log_files = sorted((data_dir / "runs").glob(f"*-{slug}.jsonl"))
    assert log_files
    return [
        json.loads(line)
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
    ]
