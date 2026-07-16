"""Filesystem helpers for inventory runs."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from uni_rag_agent.source_filters import is_ipynb_checkpoint_path

from .models import InventoryError

HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_course_entries(
    courses_root: Path,
    diagnostics: list[str],
) -> tuple[list[Path], list[Path]]:
    root_files: list[Path] = []
    course_dirs: list[Path] = []
    try:
        with os.scandir(courses_root) as entries:
            for entry in entries:
                try:
                    if is_ipynb_checkpoint_path(entry.path):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        course_dirs.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        root_files.append(Path(entry.path))
                except OSError as exc:
                    diagnostics.append(f"Could not inspect {entry.path}: {exc}")
    except OSError as exc:
        raise InventoryError(
            f"Could not list Courses root {courses_root}: {exc}"
        ) from exc

    return (
        sorted(root_files, key=lambda path: path.name.lower()),
        sorted(course_dirs, key=lambda path: path.name.lower()),
    )


def walk_files(root: Path, diagnostics: list[str]) -> Iterable[Path]:
    if is_ipynb_checkpoint_path(root):
        return

    stack = [root]
    while stack:
        current = stack.pop()
        directories: list[Path] = []
        files: list[Path] = []
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if is_ipynb_checkpoint_path(entry.path):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            directories.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            files.append(Path(entry.path))
                    except OSError as exc:
                        diagnostics.append(f"Could not inspect {entry.path}: {exc}")
        except OSError as exc:
            diagnostics.append(f"Could not list directory {current}: {exc}")
            continue

        yield from sorted(files, key=lambda path: path.name.lower())
        stack.extend(reversed(sorted(directories, key=lambda path: path.name.lower())))


def relative_path(path: Path, courses_root: Path) -> str:
    try:
        return str(path.relative_to(courses_root))
    except ValueError:
        return path.name


def timestamp_from_epoch(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
