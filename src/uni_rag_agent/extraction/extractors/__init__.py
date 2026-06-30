"""Dispatch text extraction to format-specific extractors."""

from __future__ import annotations

from uni_rag_agent.config import Config

from .._textutils import _json_dumps, _package_version
from ..models import ExtractionFailure, PendingFileRecord, RawChunk
from .code import _extract_other_code, _extract_python
from .documents import (
    _extract_docx,
    _extract_markdown,
    _extract_pdf,
    _extract_plain_text,
)
from .notebooks import _extract_notebook
from .slides import _extract_pptx
from .transcripts import _extract_vtt


def extract_raw_chunks(
    file_record: PendingFileRecord,
    config: Config,
) -> tuple[RawChunk, ...]:
    extension = file_record.extension
    path = file_record.path
    if extension == ".pdf":
        return _extract_pdf(path, config)
    if extension == ".pptx":
        return _extract_pptx(path)
    if extension == ".docx":
        return _extract_docx(path)
    if extension == ".txt":
        return _extract_plain_text(path)
    if extension == ".md":
        return _extract_markdown(path)
    if extension == ".ipynb":
        return _extract_notebook(path)
    if extension == ".py":
        return _extract_python(path)
    if extension in {".r", ".cpp", ".h", ".m"}:
        return _extract_other_code(path, extension)
    if extension == ".vtt":
        return _extract_vtt(path)
    raise ExtractionFailure(
        f"unsupported text extraction extension: {extension or '<none>'}",
        extractor_name=extractor_name_for_extension(extension),
        extractor_version=None,
        metadata_json=_json_dumps({"extension": extension}),
    )


def extractor_name_for_extension(extension: str) -> str:
    return {
        ".pdf": "pdf-pymupdf",
        ".pptx": "pptx-python-pptx",
        ".ppt": "legacy-ppt-unsupported",
        ".docx": "docx-python-docx",
        ".doc": "legacy-doc-unsupported",
        ".txt": "plain-text",
        ".md": "markdown-text",
        ".ipynb": "notebook-nbformat",
        ".py": "python-ast",
        ".r": "code-regex",
        ".cpp": "code-regex",
        ".h": "code-regex",
        ".m": "code-regex",
        ".vtt": "vtt-parser",
    }.get(extension, "unsupported-extractor")


def extractor_version_for_extension(extension: str) -> str | None:
    package_name = {
        ".pdf": "PyMuPDF",
        ".pptx": "python-pptx",
        ".docx": "python-docx",
        ".ipynb": "nbformat",
        ".doc": None,
        ".ppt": None,
    }.get(extension)
    if package_name is None:
        return None
    return _package_version(package_name)
