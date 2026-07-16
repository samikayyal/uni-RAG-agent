"""Jupyter notebook extraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import re
from typing import Any

from .._textutils import _truncate
from ..constants import NOTEBOOK_OUTPUT_CHAR_LIMIT
from ..models import RawChunk

_MARKDOWN_INLINE_DATA_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^)]*\)",
    re.IGNORECASE,
)
_HTML_INLINE_DATA_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?P<quote>['\"])"
    r"data:image/[^'\"]*(?P=quote)[^>]*>",
    re.IGNORECASE,
)


def _extract_notebook(path: Path) -> tuple[RawChunk, ...]:
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    raw_chunks: list[RawChunk] = []
    for cell_index, cell in enumerate(notebook.cells, start=1):
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code"}:
            continue
        source = str(cell.get("source", ""))
        inline_images_removed = 0
        if cell_type == "markdown":
            source, inline_images_removed = _remove_inline_image_data(source)
        source = source.strip()
        text = source
        output_text = ""
        if cell_type == "code":
            output_text = _notebook_output_text(cell.get("outputs", []))
            if output_text:
                text = f"{source}\n\nOutput:\n{output_text}".strip()
        if not text:
            continue
        raw_chunks.append(
            RawChunk(
                source_type="notebook",
                title=f"{cell_type} cell {cell_index}",
                text=text,
                location_type="notebook_cell",
                location_value=str(cell_index),
                metadata={
                    "cell": cell_index,
                    "cell_type": cell_type,
                    "inline_images_removed": inline_images_removed,
                    "output_truncated_to_chars": (
                        NOTEBOOK_OUTPUT_CHAR_LIMIT if output_text else None
                    ),
                },
            )
        )
    return tuple(raw_chunks)


def _remove_inline_image_data(text: str) -> tuple[str, int]:
    """Remove embedded image payloads while preserving useful cell text.

    Notebook markdown commonly stores rendered images as data URIs inside
    Markdown or HTML image tags. Those payloads are binary noise for semantic
    retrieval and can exceed a hosted embedding model's context window. Keep
    the Markdown alt text when available, replace HTML/data-only images with a
    short marker, and leave external image URLs untouched.
    """

    def replace_markdown(match: re.Match[str]) -> str:
        alt = re.sub(r"\s+", " ", match.group("alt")).strip()
        return alt or "[embedded image omitted]"

    cleaned, markdown_count = _MARKDOWN_INLINE_DATA_IMAGE_RE.subn(
        replace_markdown,
        text,
    )
    cleaned, html_count = _HTML_INLINE_DATA_IMAGE_RE.subn(
        "[embedded image omitted]",
        cleaned,
    )
    cleaned = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", cleaned)
    return cleaned, markdown_count + html_count


def _notebook_output_text(outputs: Iterable[Mapping[str, Any]]) -> str:
    pieces: list[str] = []
    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "stream":
            text = output.get("text", "")
            pieces.append(_stringify_notebook_text(text))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if isinstance(data, Mapping) and "text/plain" in data:
                pieces.append(_stringify_notebook_text(data["text/plain"]))
        elif output_type == "error":
            trace_lines = output.get("traceback", [])
            pieces.append(_stringify_notebook_text(trace_lines))

    text = "\n".join(piece for piece in pieces if piece.strip()).strip()
    return _truncate(text, NOTEBOOK_OUTPUT_CHAR_LIMIT)


def _stringify_notebook_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)
