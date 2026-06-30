"""Transcript extraction."""

from __future__ import annotations

from pathlib import Path

from .._textutils import _read_text_file
from ..constants import VTT_TIMESTAMP_RE
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
