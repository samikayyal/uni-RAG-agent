from __future__ import annotations

import os
import shutil
from contextlib import closing
from pathlib import Path

import pytest

from uni_rag_agent.config import Config, load_config
from uni_rag_agent.inventory import (
    InventoryError,
    MISSING_REASON,
    classify_file,
    inventory_courses,
    load_inventory_summary,
)
from uni_rag_agent.inventory import core as inventory_core
from uni_rag_agent.inventory import file_io as inventory_file_io
from uni_rag_agent.storage import connect_sqlite


def make_config(tmp_path: Path) -> Config:
    (tmp_path / "Courses").mkdir()
    return load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")


@pytest.mark.parametrize(
    ("filename", "category", "index_status", "has_reason"),
    [
        ("lecture.pdf", "document", "pending", False),
        ("slides.pptx", "slides", "pending", False),
        ("notebook.ipynb", "notebook", "pending", False),
        ("assignment.py", "code", "pending", False),
        ("dataset.csv", "data_schema", "pending", False),
        ("captions.vtt", "transcript", "pending", False),
        ("diagram.PNG", "image_metadata_only", "metadata_only", True),
        ("archive.zip", "archive_metadata_only", "metadata_only", True),
        ("setup.exe", "installer_metadata_only", "metadata_only", True),
        ("vectors.bin", "model_metadata_only", "metadata_only", True),
        ("unknown", "unknown_metadata_only", "metadata_only", True),
    ],
    ids=[
        "pdf-document",
        "pptx-slides",
        "ipynb-notebook",
        "py-code",
        "csv-data-schema",
        "vtt-transcript",
        "png-image-metadata-only",
        "zip-archive-metadata-only",
        "exe-installer-metadata-only",
        "bin-model-metadata-only",
        "no-extension-unknown-metadata-only",
    ],
)
def test_classify_file_uses_spec_category_status_and_reason(
    filename: str,
    category: str,
    index_status: str,
    has_reason: bool,
) -> None:
    classification = classify_file(Path(filename))

    assert classification.category == category
    assert classification.index_status == index_status
    assert (classification.reason_not_indexed is not None) is has_reason


def test_inventory_run_preserves_exact_paths_and_classifies_mixed_files(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "High Preformance Computing for Big Data"
    nested_dir = course_dir / "Week 01"
    nested_dir.mkdir(parents=True)
    (course_dir / "lecture_notes.md").write_text("# MapReduce", encoding="utf-8")
    (nested_dir / "diagram.png").write_bytes(b"png")
    (course_dir / "lecture.mp4").write_bytes(b"mp4")
    (course_dir / "archive.zip").write_bytes(b"zip")
    (course_dir / "setup.exe").write_bytes(b"exe")
    (course_dir / "vectors.bin").write_bytes(b"bin")
    (course_dir / "legacy.doc").write_text("legacy", encoding="utf-8")
    (course_dir / "unknown").write_text("unknown", encoding="utf-8")

    result = inventory_courses(config)

    assert result.status == "completed"
    assert result.courses_seen == 1
    assert result.files_seen == 8
    assert result.files_pending == 2
    assert result.files_metadata_only == 6
    assert result.files_failed == 0
    assert result.by_category["document"] == 2
    assert result.by_category["image_metadata_only"] == 1
    assert result.by_category["media_metadata_only"] == 1
    assert result.by_category["archive_metadata_only"] == 1
    assert result.by_category["installer_metadata_only"] == 1
    assert result.by_category["model_metadata_only"] == 1
    assert result.by_category["unknown_metadata_only"] == 1

    with closing(connect_sqlite(config)) as connection:
        course_row = connection.execute(
            "SELECT name, path, file_count, total_bytes FROM courses"
        ).fetchone()
        file_rows = connection.execute(
            """
            SELECT relative_path, filename, extension, category, index_status,
                   reason_not_indexed
            FROM files
            ORDER BY relative_path
            """
        ).fetchall()

    assert course_row["name"] == "High Preformance Computing for Big Data"
    assert course_row["path"] == str(course_dir)
    assert course_row["file_count"] == 8

    rows_by_name = {row["filename"]: row for row in file_rows}
    assert rows_by_name["lecture_notes.md"]["relative_path"] == str(
        Path("High Preformance Computing for Big Data") / "lecture_notes.md"
    )
    assert rows_by_name["lecture_notes.md"]["extension"] == ".md"
    assert rows_by_name["lecture_notes.md"]["category"] == "document"
    assert rows_by_name["lecture_notes.md"]["index_status"] == "pending"
    assert rows_by_name["lecture_notes.md"]["reason_not_indexed"] is None
    assert rows_by_name["legacy.doc"]["category"] == "document"
    assert rows_by_name["legacy.doc"]["index_status"] == "pending"
    assert rows_by_name["diagram.png"]["index_status"] == "metadata_only"
    assert "standalone image" in rows_by_name["diagram.png"]["reason_not_indexed"]
    assert "transcription" in rows_by_name["lecture.mp4"]["reason_not_indexed"]
    assert "not decompressed" in rows_by_name["archive.zip"]["reason_not_indexed"]
    assert "never executed" in rows_by_name["setup.exe"]["reason_not_indexed"]
    assert "serialized artifact" in rows_by_name["vectors.bin"]["reason_not_indexed"]
    assert "unsupported extension" in rows_by_name["unknown"]["reason_not_indexed"]


def test_inventory_rerun_is_idempotent_and_skips_hash_for_unchanged_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    source_file = course_dir / "syllabus.txt"
    source_file.write_text("BM25 and vector search", encoding="utf-8")

    original_hash = inventory_core.sha256_file
    hashed_paths: list[Path] = []

    def tracking_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_hash(path)

    monkeypatch.setattr(inventory_core, "sha256_file", tracking_hash)

    first = inventory_courses(config)
    assert first.files_seen == 1
    assert hashed_paths == [source_file]

    hashed_paths.clear()
    second = inventory_courses(config)
    assert second.files_seen == 1
    assert second.files_missing == 0
    assert hashed_paths == []

    with closing(connect_sqlite(config)) as connection:
        row_count = connection.execute("SELECT COUNT(*) AS count FROM files").fetchone()
        run_count = connection.execute(
            "SELECT COUNT(*) AS count FROM extraction_runs"
        ).fetchone()

    assert row_count["count"] == 1
    assert run_count["count"] == 2


def test_changed_timestamp_triggers_hash_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    source_file = course_dir / "syllabus.txt"
    source_file.write_text("original", encoding="utf-8")
    inventory_courses(config)

    original_hash = inventory_core.sha256_file
    hashed_paths: list[Path] = []

    def tracking_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_hash(path)

    monkeypatch.setattr(inventory_core, "sha256_file", tracking_hash)
    source_file.write_text("changed", encoding="utf-8")
    current_mtime = source_file.stat().st_mtime
    os.utime(source_file, (current_mtime + 5, current_mtime + 5))

    result = inventory_courses(config)

    assert result.files_seen == 1
    assert hashed_paths == [source_file]


def test_missing_file_is_marked_without_hard_delete(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    source_file = course_dir / "syllabus.txt"
    source_file.write_text("BM25", encoding="utf-8")
    inventory_courses(config)

    source_file.unlink()
    result = inventory_courses(config)

    assert result.files_seen == 0
    assert result.files_missing == 1

    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            "SELECT index_status, reason_not_indexed FROM files WHERE filename = ?",
            ("syllabus.txt",),
        ).fetchone()

    assert row["index_status"] == "skipped"
    assert row["reason_not_indexed"] == MISSING_REASON


def test_removed_course_folder_resets_current_course_totals(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    source_file = course_dir / "syllabus.txt"
    source_file.write_text("BM25", encoding="utf-8")
    inventory_courses(config)

    shutil.rmtree(course_dir)
    result = inventory_courses(config)
    summary = load_inventory_summary(config)

    assert result.courses_seen == 0
    assert result.files_seen == 0
    assert result.files_missing == 1

    assert len(summary.by_course) == 1
    course_summary = summary.by_course[0]
    assert course_summary.name == "Information Retrieval"
    assert course_summary.file_count == 0
    assert course_summary.total_bytes == 0

    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            "SELECT index_status, reason_not_indexed FROM files WHERE filename = ?",
            ("syllabus.txt",),
        ).fetchone()

    assert row["index_status"] == "skipped"
    assert row["reason_not_indexed"] == MISSING_REASON


def test_inventory_run_record_does_not_report_pending_files_as_indexed(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    (course_dir / "syllabus.txt").write_text("BM25", encoding="utf-8")
    (course_dir / "diagram.png").write_bytes(b"png")

    result = inventory_courses(config)

    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            """
            SELECT files_seen, files_indexed, files_metadata_only, files_failed
            FROM extraction_runs
            WHERE id = ?
            """,
            (result.run_id,),
        ).fetchone()

    assert result.files_pending == 1
    assert row["files_seen"] == 2
    assert row["files_indexed"] == 0
    assert row["files_metadata_only"] == 1
    assert row["files_failed"] == 0


def test_inventory_apis_close_sqlite_connections(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    (course_dir / "syllabus.txt").write_text("BM25", encoding="utf-8")

    inventory_courses(config)
    load_inventory_summary(config)

    config.sqlite_path.unlink()
    assert not config.sqlite_path.exists()


def test_inventory_summary_reports_current_database_counts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    (course_dir / "syllabus.txt").write_text("BM25", encoding="utf-8")
    (course_dir / "diagram.png").write_bytes(b"png")
    inventory_courses(config)

    summary = load_inventory_summary(config)

    assert summary.courses_total == 1
    assert summary.files_total == 2
    assert summary.files_missing == 0
    assert summary.latest_inventory_run_id is not None
    assert summary.by_status["pending"] == 1
    assert summary.by_status["metadata_only"] == 1
    assert summary.by_category["document"] == 1
    assert summary.by_category["image_metadata_only"] == 1


def test_stat_failure_marks_file_failed_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    unreadable_file = course_dir / "broken.txt"
    readable_file = course_dir / "syllabus.txt"
    unreadable_file.write_text("cannot inspect metadata", encoding="utf-8")
    readable_file.write_text("BM25", encoding="utf-8")
    original_stat = Path.stat

    def failing_stat(self: Path, *args, **kwargs):
        if self == unreadable_file:
            raise OSError("metadata denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)

    result = inventory_courses(config)

    assert result.status == "completed"
    assert result.files_seen == 2
    assert result.files_failed == 1
    assert result.files_pending == 1
    assert any("Could not stat file" in item for item in result.diagnostics)

    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute(
            """
            SELECT filename, size_bytes, index_status, reason_not_indexed
            FROM files
            ORDER BY filename
            """
        ).fetchall()
        run = connection.execute(
            """
            SELECT status, files_seen, files_failed
            FROM extraction_runs
            WHERE id = ?
            """,
            (result.run_id,),
        ).fetchone()

    rows_by_name = {row["filename"]: row for row in rows}
    assert rows_by_name["broken.txt"]["size_bytes"] == 0
    assert rows_by_name["broken.txt"]["index_status"] == "failed"
    assert "metadata read failed" in rows_by_name["broken.txt"]["reason_not_indexed"]
    assert rows_by_name["syllabus.txt"]["index_status"] == "pending"
    assert run["status"] == "completed"
    assert run["files_seen"] == 2
    assert run["files_failed"] == 1


def test_hash_failure_marks_file_failed_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    unreadable_file = course_dir / "broken.txt"
    readable_file = course_dir / "syllabus.txt"
    unreadable_file.write_text("cannot hash", encoding="utf-8")
    readable_file.write_text("BM25", encoding="utf-8")
    original_hash = inventory_core.sha256_file

    def failing_hash(path: Path) -> str:
        if path == unreadable_file:
            raise OSError("hash denied")
        return original_hash(path)

    monkeypatch.setattr(inventory_core, "sha256_file", failing_hash)

    result = inventory_courses(config)

    assert result.status == "completed"
    assert result.files_seen == 2
    assert result.files_failed == 1
    assert result.files_pending == 1
    assert any("Could not hash file" in item for item in result.diagnostics)

    with closing(connect_sqlite(config)) as connection:
        rows = connection.execute(
            """
            SELECT filename, content_hash, index_status, reason_not_indexed
            FROM files
            ORDER BY filename
            """
        ).fetchall()
        run = connection.execute(
            """
            SELECT status, files_seen, files_failed
            FROM extraction_runs
            WHERE id = ?
            """,
            (result.run_id,),
        ).fetchone()

    rows_by_name = {row["filename"]: row for row in rows}
    assert rows_by_name["broken.txt"]["content_hash"] is None
    assert rows_by_name["broken.txt"]["index_status"] == "failed"
    assert "hashing failed" in rows_by_name["broken.txt"]["reason_not_indexed"]
    assert rows_by_name["syllabus.txt"]["index_status"] == "pending"
    assert run["status"] == "completed"
    assert run["files_seen"] == 2
    assert run["files_failed"] == 1


def test_hash_failure_recovers_when_later_hash_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    course_dir.mkdir()
    source_file = course_dir / "syllabus.txt"
    source_file.write_text("BM25", encoding="utf-8")
    original_hash = inventory_core.sha256_file

    def failing_hash(path: Path) -> str:
        if path == source_file:
            raise OSError("hash denied")
        return original_hash(path)

    monkeypatch.setattr(inventory_core, "sha256_file", failing_hash)
    first = inventory_courses(config)

    monkeypatch.setattr(inventory_core, "sha256_file", original_hash)
    second = inventory_courses(config)

    assert first.files_failed == 1
    assert second.files_failed == 0
    assert second.files_pending == 1

    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            """
            SELECT content_hash, index_status, reason_not_indexed
            FROM files
            WHERE filename = ?
            """,
            ("syllabus.txt",),
        ).fetchone()

    assert row["content_hash"] is not None
    assert row["index_status"] == "pending"
    assert row["reason_not_indexed"] is None


def test_nested_directory_listing_failure_records_diagnostic_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    course_dir = config.courses_root / "Information Retrieval"
    blocked_dir = course_dir / "Blocked"
    course_dir.mkdir()
    blocked_dir.mkdir()
    (course_dir / "syllabus.txt").write_text("BM25", encoding="utf-8")
    (blocked_dir / "hidden.txt").write_text("hidden", encoding="utf-8")
    original_scandir = inventory_file_io.os.scandir

    def failing_scandir(path):
        if Path(path) == blocked_dir:
            raise OSError("directory denied")
        return original_scandir(path)

    monkeypatch.setattr(inventory_file_io.os, "scandir", failing_scandir)

    result = inventory_courses(config)

    assert result.status == "completed"
    assert result.files_seen == 1
    assert result.files_failed == 0
    assert any("Could not list directory" in item for item in result.diagnostics)

    with closing(connect_sqlite(config)) as connection:
        filenames = [
            row["filename"]
            for row in connection.execute(
                "SELECT filename FROM files ORDER BY filename"
            ).fetchall()
        ]
        course_row = connection.execute(
            "SELECT file_count FROM courses WHERE name = ?",
            ("Information Retrieval",),
        ).fetchone()

    assert filenames == ["syllabus.txt"]
    assert course_row["file_count"] == 1


def test_courses_root_listing_failure_records_failed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    original_scandir = inventory_file_io.os.scandir

    def failing_scandir(path):
        if Path(path) == config.courses_root:
            raise OSError("root denied")
        return original_scandir(path)

    monkeypatch.setattr(inventory_file_io.os, "scandir", failing_scandir)

    with pytest.raises(InventoryError, match="Could not list Courses root"):
        inventory_courses(config)

    with closing(connect_sqlite(config)) as connection:
        row = connection.execute(
            """
            SELECT status, finished_at, files_seen, files_failed, error
            FROM extraction_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    assert row["status"] == "failed"
    assert row["finished_at"] is not None
    assert row["files_seen"] == 0
    assert row["files_failed"] == 0
    assert "Could not list Courses root" in row["error"]
