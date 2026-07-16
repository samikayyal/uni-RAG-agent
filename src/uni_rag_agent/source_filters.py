"""Shared source-file exclusion rules for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

IPYNB_CHECKPOINTS_DIR_NAME = ".ipynb_checkpoints"


def is_ipynb_checkpoint_path(path: Path | str) -> bool:
    """Return whether *path* is inside a Jupyter checkpoint directory.

    Jupyter creates these directories at arbitrary depths below a notebook.
    Matching a path component, rather than only a suffix, excludes the whole
    checkpoint subtree while leaving ordinary directories named ``checkpoints``
    alone.
    """

    parts = str(path).replace("\\", "/").split("/")
    return any(part.casefold() == IPYNB_CHECKPOINTS_DIR_NAME for part in parts if part)


__all__ = [
    "IPYNB_CHECKPOINTS_DIR_NAME",
    "is_ipynb_checkpoint_path",
]
