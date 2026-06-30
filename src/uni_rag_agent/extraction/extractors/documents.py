"""Document and plain-text extractors."""

from __future__ import annotations

import io
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from uni_rag_agent.config import Config

from .._textutils import (
    _count_tokens,
    _json_dumps,
    _package_version,
    _read_text_file,
    _title_from_text,
)
from ..constants import (
    DEFAULT_MAX_CHUNK_TOKENS,
    PDF_SCANNED_TEXT_CHAR_THRESHOLD,
    SCANNED_PDF_OCR_REASON,
)
from ..models import ExtractionFailure, RawChunk


def _extract_pdf(path: Path, config: Config) -> tuple[RawChunk, ...]:
    import fitz

    raw_chunks: list[RawChunk] = []
    page_count = 0
    with fitz.open(path) as document:
        page_count = document.page_count
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                raw_chunks.append(
                    RawChunk(
                        source_type="document",
                        title=_title_from_text(text),
                        text=text,
                        location_type="page",
                        location_value=str(page_index),
                        metadata={"page": page_index},
                    )
                )

        total_text = "\n".join(chunk.text for chunk in raw_chunks)
        if page_count and len(total_text.strip()) < PDF_SCANNED_TEXT_CHAR_THRESHOLD:
            if not config.ocr_enabled:
                raise ExtractionFailure(
                    SCANNED_PDF_OCR_REASON,
                    extractor_name="pdf-pymupdf",
                    extractor_version=_package_version("PyMuPDF"),
                    metadata_json=_json_dumps(
                        {
                            "page_count": page_count,
                            "ocr_enabled": False,
                        }
                    ),
                )
            raw_chunks = _ocr_pdf_pages(document)

    return tuple(raw_chunks)


def _ocr_pdf_pages(document: Any) -> list[RawChunk]:
    if shutil.which("tesseract") is None:
        raise ExtractionFailure(
            SCANNED_PDF_OCR_REASON,
            extractor_name="pdf-pymupdf-ocr",
            extractor_version=_package_version("pytesseract"),
            metadata_json=_json_dumps({"tesseract_available": False}),
        )

    import fitz
    import pytesseract
    from PIL import Image

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionFailure(
            SCANNED_PDF_OCR_REASON,
            extractor_name="pdf-pymupdf-ocr",
            extractor_version=_package_version("pytesseract"),
            metadata_json=_json_dumps({"tesseract_available": False}),
        ) from exc

    raw_chunks: list[RawChunk] = []
    matrix = fitz.Matrix(2, 2)
    for page_index, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        text = pytesseract.image_to_string(image).strip()
        if text:
            raw_chunks.append(
                RawChunk(
                    source_type="document",
                    title=_title_from_text(text),
                    text=text,
                    location_type="page",
                    location_value=str(page_index),
                    metadata={"page": page_index, "ocr": True},
                )
            )
    return raw_chunks


def _extract_docx(path: Path) -> tuple[RawChunk, ...]:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(path)
    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            text = Paragraph(child, document).text.strip()
            if text:
                blocks.append(text)
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            rows: list[str] = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            table_text = "\n".join(row for row in rows if row.strip()).strip()
            if table_text:
                blocks.append(table_text)

    return tuple(
        _group_text_blocks(
            blocks,
            source_type="document",
            location_type="docx_section",
            title_prefix="DOCX section",
            metadata={"format": "docx"},
        )
    )


def _extract_plain_text(path: Path) -> tuple[RawChunk, ...]:
    text = _read_text_file(path)
    blocks = _paragraph_blocks(text)
    return tuple(
        _group_text_blocks(
            blocks,
            source_type="document",
            location_type="text_section",
            title_prefix="Text section",
            metadata={"format": "text"},
        )
    )


def _extract_markdown(path: Path) -> tuple[RawChunk, ...]:
    text = _read_text_file(path)
    sections = _markdown_sections(text)
    raw_chunks: list[RawChunk] = []
    for section_index, (title, section_text) in enumerate(sections, start=1):
        if not section_text.strip():
            continue
        raw_chunks.append(
            RawChunk(
                source_type="document",
                title=title or _title_from_text(section_text),
                text=section_text.strip(),
                location_type="markdown_section",
                location_value=str(section_index),
                metadata={"section": section_index, "format": "markdown"},
            )
        )
    if raw_chunks:
        return tuple(raw_chunks)
    return _extract_plain_text(path)


def _paragraph_blocks(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    return [block.strip() for block in blocks if block.strip()]


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if heading and current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))
            current_lines = []
        if heading:
            current_title = heading.group(2).strip()
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def _group_text_blocks(
    blocks: Iterable[str],
    *,
    source_type: str,
    location_type: str,
    title_prefix: str,
    metadata: Mapping[str, object],
) -> list[RawChunk]:
    raw_chunks: list[RawChunk] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_tokens = _count_tokens(block)
        if current and current_tokens + block_tokens > DEFAULT_MAX_CHUNK_TOKENS:
            section_index = len(raw_chunks) + 1
            text = "\n\n".join(current).strip()
            raw_chunks.append(
                RawChunk(
                    source_type=source_type,
                    title=f"{title_prefix} {section_index}",
                    text=text,
                    location_type=location_type,
                    location_value=str(section_index),
                    metadata={**metadata, "section": section_index},
                )
            )
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens

    if current:
        section_index = len(raw_chunks) + 1
        text = "\n\n".join(current).strip()
        raw_chunks.append(
            RawChunk(
                source_type=source_type,
                title=f"{title_prefix} {section_index}",
                text=text,
                location_type=location_type,
                location_value=str(section_index),
                metadata={**metadata, "section": section_index},
            )
        )
    return raw_chunks
