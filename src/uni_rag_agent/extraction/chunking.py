"""Convert natural extraction units into persisted chunk records."""

from __future__ import annotations

from collections.abc import Iterable

from ._textutils import _count_tokens, _json_dumps
from .models import ChunkRecord, PendingFileRecord, RawChunk


def finalize_chunks(
    *,
    file_record: PendingFileRecord,
    raw_chunks: Iterable[RawChunk],
    max_tokens: int,
) -> tuple[ChunkRecord, ...]:
    chunks: list[ChunkRecord] = []
    for raw_chunk in raw_chunks:
        text = raw_chunk.text.strip()
        if not text:
            continue
        pieces = _split_text_by_tokens(text, max_tokens)
        for piece_index, piece in enumerate(pieces, start=1):
            is_subchunk = len(pieces) > 1
            metadata_payload = dict(raw_chunk.metadata)
            metadata_payload.update(
                {
                    "source_location_type": raw_chunk.location_type,
                    "source_location_value": raw_chunk.location_value,
                }
            )
            if is_subchunk:
                metadata_payload.update(
                    {
                        "subchunk_index": piece_index,
                        "subchunk_count": len(pieces),
                    }
                )
                location_type = "subchunk"
                location_value = (
                    f"{raw_chunk.location_type}:{raw_chunk.location_value}:"
                    f"part:{piece_index}"
                )
            else:
                location_type = raw_chunk.location_type
                location_value = raw_chunk.location_value

            chunk_index = len(chunks)
            chunks.append(
                ChunkRecord(
                    file_id=file_record.id,
                    chunk_uid=f"file-{file_record.id}-chunk-{chunk_index}",
                    source_type=raw_chunk.source_type,
                    chunk_index=chunk_index,
                    title=raw_chunk.title,
                    text=piece,
                    token_count=_count_tokens(piece),
                    location_type=location_type,
                    location_value=location_value,
                    metadata_json=_json_dumps(metadata_payload),
                )
            )
    return tuple(chunks)


def _split_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    return [
        " ".join(words[index : index + max_tokens]).strip()
        for index in range(0, len(words), max_tokens)
    ]
