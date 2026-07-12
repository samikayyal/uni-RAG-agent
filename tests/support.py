from __future__ import annotations

import dataclasses
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from uni_rag_agent.config import Config, load_config
from uni_rag_agent.storage import connect_sqlite, ensure_data_dirs, initialize_schema

UNI_RAG_ENV_PREFIX = "UNI_RAG_"


def make_config(tmp_path: Path, **overrides: object) -> Config:
    """Build an isolated config while leaving scenario data setup to the test."""
    (tmp_path / "Courses").mkdir(parents=True, exist_ok=True)
    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    return dataclasses.replace(config, **overrides)


def make_initialized_config(tmp_path: Path, **overrides: object) -> Config:
    """Build a config and initialize its generated directories and SQLite schema."""
    config = make_config(tmp_path, **overrides)
    with initialized_connection(config):
        pass
    return config


@contextmanager
def initialized_connection(config: Config) -> Iterator[sqlite3.Connection]:
    """Open a writable test connection after applying the shared schema policy."""
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        yield connection


def clean_subprocess_env(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the host environment without inherited project configuration."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(UNI_RAG_ENV_PREFIX)
    }
    if overrides:
        env.update(overrides)
    return env
