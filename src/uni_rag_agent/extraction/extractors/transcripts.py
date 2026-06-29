"""Transcript extraction."""

from __future__ import annotations

from pathlib import Path

from ..constants import TEXT_ENCODINGS, VTT_TIMESTAMP_RE
from ..models import RawChunk


def _extract_vtt(path: Path) -> tuple[RawChunk, ...]:
    text = _read_text_file(path)
    lines = text.splitlines()
    raw_chunks: list[RawChunk] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        match = VTT_TIMESTAMP_RE.match(line)
        if not match:
            index += 1
            continue
        start = match.group("start")
        end = match.group("end")
        index += 1
        cue_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            cue_lines.append(lines[index].strip())
            index += 1
        cue_text = "\n".join(cue_lines).strip()
        if cue_text:
            raw_chunks.append(
                RawChunk(
                    source_type="transcript",
                    title=start,
                    text=cue_text,
                    location_type="timestamp",
                    location_value=start,
                    metadata={"timestamp_start": start, "timestamp_end": end},
                )
            )
    return tuple(raw_chunks)


def _read_text_file(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return path.read_text()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
