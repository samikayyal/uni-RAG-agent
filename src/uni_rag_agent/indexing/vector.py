"""ChromaDB vector indexing and semantic search.

SQLite stays authoritative for chunk text and citation metadata. ChromaDB holds
only vectors plus filter metadata. Embedding mapping rows in the SQLite
``embeddings`` table record the physical collection and stable vector id so the
two stores can be reconciled. Different embedding models persist into distinct
model-namespaced physical collections, enabling side-by-side models.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone

from uni_rag_agent.config import Config
from uni_rag_agent.retrieval import RetrievalResult
from uni_rag_agent.storage import (
    StorageError,
    check_storage,
    connect_sqlite,
    connect_sqlite_read_only,
    ensure_data_dirs,
    initialize_schema,
)

from .eligibility import (
    INDEX_TO_SOURCE_TYPE,
    current_chunk_where_sql,
    placeholders,
    source_types_for_indexes,
    validate_logical_index,
)
from .embeddings import build_embedding_model
from .models import SemanticSearchError, VectorIndexError, VectorIndexResult
from .profiles import EmbeddingProfile, physical_collection_name

_EMBED_BATCH = 64
_SNIPPET_CHAR_LIMIT = 240
VECTOR_BACKEND = "chroma"

_INSERT_EMBEDDING_SQL = """
INSERT INTO embeddings (
    chunk_id,
    vector_backend,
    vector_collection,
    vector_id,
    embedding_model,
    embedding_dim,
    embedded_at
)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chunk_id, embedding_model) DO UPDATE SET
    vector_backend = excluded.vector_backend,
    vector_collection = excluded.vector_collection,
    vector_id = excluded.vector_id,
    embedding_dim = excluded.embedding_dim,
    embedded_at = excluded.embedded_at
"""


@dataclass(frozen=True)
class _Candidate:
    chunk_id: int
    distance: float
    physical_collection: str
    vector_id: str


def sync_vector_index(
    config: Config,
    collection: str | None = None,
    model: str | None = None,
    rebuild: bool = False,
) -> VectorIndexResult:
    """Embed eligible chunks into ChromaDB for the selected model.

    The default behavior is incremental: only current eligible chunks that are
    missing an embedding for the selected model are embedded. ``rebuild`` clears
    and repopulates only the selected model/profile (and optional logical
    ``collection``).
    """
    built = build_embedding_model(config, model, error=VectorIndexError)
    profile = built.profile
    dimension = built.dimension
    selected = _selected_logical_indexes(collection)

    ensure_data_dirs(config)
    try:
        client = _chroma_client(config, error=VectorIndexError)
        rows_removed = 0
        vectors_indexed = 0
        by_source_type: dict[str, int] = {}
        physical_names: list[str] = []
        with closing(connect_sqlite(config)) as connection:
            initialize_schema(connection)
            for logical_index, source_type in selected:
                physical = _physical_name(logical_index, profile, dimension)
                physical_names.append(physical)
                if rebuild:
                    rows_removed += _clear_collection(
                        client,
                        connection,
                        physical=physical,
                        model_name=profile.model_name,
                    )
                chroma_collection = client.get_or_create_collection(
                    name=physical,
                    metadata={"hnsw:space": profile.metric},
                )
                indexed = _embed_missing_chunks(
                    connection,
                    chroma_collection=chroma_collection,
                    source_type=source_type,
                    logical_index=logical_index,
                    physical=physical,
                    model_name=profile.model_name,
                    dimension=dimension,
                    embeddings=built.embeddings,
                    course_field=True,
                )
                if indexed:
                    by_source_type[source_type] = indexed
                    vectors_indexed += indexed
            chunks_seen = _eligible_chunk_count(
                connection, [source_type for _, source_type in selected]
            )
            embeddings_total = _embeddings_total(
                connection,
                model_name=profile.model_name,
                physical_names=physical_names,
            )
            connection.commit()
    except VectorIndexError:
        raise
    except sqlite3.Error as exc:
        raise VectorIndexError(f"Vector index sync failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface ChromaDB failures clearly
        raise VectorIndexError(f"Vector index sync failed: {exc}") from exc

    diagnostics = _sync_diagnostics(
        chunks_seen=chunks_seen,
        vectors_indexed=vectors_indexed,
        embeddings_total=embeddings_total,
        model_name=profile.model_name,
    )
    return VectorIndexResult(
        rebuild=rebuild,
        model=profile.model_name,
        provider=profile.provider,
        embedding_dim=dimension,
        collections=tuple(logical for logical, _ in selected),
        chunks_seen=chunks_seen,
        rows_removed=rows_removed,
        vectors_indexed=vectors_indexed,
        embeddings_total=embeddings_total,
        by_source_type=dict(sorted(by_source_type.items())),
        diagnostics=tuple(diagnostics),
    )


def semantic_search(
    config: Config,
    query: str,
    course: str | None = None,
    indexes: Sequence[str] | None = None,
    top_k: int | None = None,
    model: str | None = None,
) -> list[RetrievalResult]:
    """Run semantic vector search over the selected model's collections.

    Mirrors ``keyword_search(config, query, ...)``. Queries the selected
    physical ChromaDB collections, then joins candidate ids back to SQLite in
    read-only mode and reapplies the current-file-only, course, and logical
    index filters. Returns ``[]`` when nothing matches. Does not persist
    ``search_runs`` or ``search_results``.
    """
    limit = top_k if top_k is not None else config.semantic_top_k
    if limit <= 0:
        raise SemanticSearchError("top_k must be greater than zero")

    query_text = query.strip()
    if not query_text:
        raise SemanticSearchError("Semantic query must not be empty.")

    source_types = source_types_for_indexes(indexes, error=SemanticSearchError)
    if source_types == ():
        return []
    selected = _selected_logical_indexes_for_search(source_types)

    built = build_embedding_model(config, model, error=SemanticSearchError)
    profile = built.profile
    dimension = built.dimension

    storage = check_storage(config)
    if not storage.ok:
        details = "; ".join(storage.diagnostics) or "storage is not ready"
        raise SemanticSearchError(f"Semantic search storage check failed: {details}")

    try:
        query_vector = built.embeddings.embed_query(query_text)
        client = _chroma_client(config, error=SemanticSearchError)
        candidates = _query_candidates(
            client,
            selected=selected,
            profile=profile,
            dimension=dimension,
            query_vector=query_vector,
            limit=limit,
        )
        if not candidates:
            return []
        rows = _hydrate_candidates(
            config,
            candidate_ids=list(candidates),
            source_types=[source_type for _, source_type in selected],
            course=course,
        )
    except SemanticSearchError:
        raise
    except sqlite3.OperationalError as exc:
        raise SemanticSearchError(
            f"Semantic query could not be executed: {exc}"
        ) from exc
    except sqlite3.Error as exc:
        raise StorageError(f"Semantic search could not inspect SQLite: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface ChromaDB failures clearly
        raise SemanticSearchError(f"Semantic search failed: {exc}") from exc

    ranked = sorted(
        rows,
        key=lambda row: (
            candidates[int(row["chunk_id"])].distance,
            int(row["chunk_id"]),
        ),
    )
    results: list[RetrievalResult] = []
    for rank, row in enumerate(ranked[:limit], start=1):
        candidate = candidates[int(row["chunk_id"])]
        results.append(
            RetrievalResult(
                chunk_id=int(row["chunk_id"]),
                file_id=int(row["file_id"]),
                course=row["course"],
                file_path=row["file_path"],
                source_type=row["source_type"],
                location_type=row["location_type"],
                location_value=row["location_value"],
                rank=rank,
                score=1.0 - float(candidate.distance),
                snippet=_snippet(row["text"]),
                retrieval_method="semantic",
                vector_collection=candidate.physical_collection,
                vector_id=candidate.vector_id,
            )
        )
    return results


def _selected_logical_indexes(collection: str | None) -> list[tuple[str, str]]:
    if collection is None:
        return list(INDEX_TO_SOURCE_TYPE.items())
    source_type = validate_logical_index(collection, error=VectorIndexError)
    return [(collection, source_type)]


def _selected_logical_indexes_for_search(
    source_types: tuple[str, ...] | None,
) -> list[tuple[str, str]]:
    if source_types is None:
        return list(INDEX_TO_SOURCE_TYPE.items())
    wanted = set(source_types)
    return [
        (logical_index, source_type)
        for logical_index, source_type in INDEX_TO_SOURCE_TYPE.items()
        if source_type in wanted
    ]


def _physical_name(
    logical_index: str,
    profile: EmbeddingProfile,
    dimension: int,
) -> str:
    return physical_collection_name(
        logical_index,
        provider=profile.provider,
        model_name=profile.model_name,
        dimension=dimension,
        metric=profile.metric,
    )


def _clear_collection(
    client: object,
    connection: sqlite3.Connection,
    *,
    physical: str,
    model_name: str,
) -> int:
    if physical in _existing_collection_names(client):
        client.delete_collection(name=physical)  # type: ignore[attr-defined]
    cursor = connection.execute(
        """
        DELETE FROM embeddings
        WHERE vector_backend = ?
          AND vector_collection = ?
          AND embedding_model = ?
        """,
        (VECTOR_BACKEND, physical, model_name),
    )
    connection.commit()
    return max(cursor.rowcount, 0)


def _embed_missing_chunks(
    connection: sqlite3.Connection,
    *,
    chroma_collection: object,
    source_type: str,
    logical_index: str,
    physical: str,
    model_name: str,
    dimension: int,
    embeddings: object,
    course_field: bool,
) -> int:
    rows = _missing_chunk_rows(
        connection, source_type=source_type, model_name=model_name
    )
    if not rows:
        return 0

    indexed = 0
    embedded_at = _utc_now()
    for batch in _batches(rows, _EMBED_BATCH):
        texts = [str(row["text"]) for row in batch]
        vectors = embeddings.embed_documents(texts)  # type: ignore[attr-defined]
        ids = [f"chunk:{int(row['chunk_id'])}" for row in batch]
        metadatas = [
            _chunk_metadata(row, logical_index, model_name, dimension) for row in batch
        ]
        chroma_collection.upsert(  # type: ignore[attr-defined]
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
        )
        for row, vector_id in zip(batch, ids):
            connection.execute(
                _INSERT_EMBEDDING_SQL,
                (
                    int(row["chunk_id"]),
                    VECTOR_BACKEND,
                    physical,
                    vector_id,
                    model_name,
                    dimension,
                    embedded_at,
                ),
            )
        connection.commit()
        indexed += len(batch)
    return indexed


def _missing_chunk_rows(
    connection: sqlite3.Connection,
    *,
    source_type: str,
    model_name: str,
) -> list[sqlite3.Row]:
    where_sql = current_chunk_where_sql((source_type,), require_non_empty_text=True)
    return connection.execute(
        f"""
        SELECT
            chunks.id AS chunk_id,
            chunks.text AS text,
            chunks.file_id AS file_id,
            chunks.source_type AS source_type,
            files.path AS file_path,
            courses.name AS course
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        LEFT JOIN courses ON courses.id = files.course_id
        WHERE {where_sql}
          AND NOT EXISTS (
              SELECT 1
              FROM embeddings
              WHERE embeddings.chunk_id = chunks.id
                AND embeddings.embedding_model = ?
          )
        ORDER BY chunks.id
        """,
        (source_type, model_name),
    ).fetchall()


def _chunk_metadata(
    row: sqlite3.Row,
    logical_index: str,
    model_name: str,
    dimension: int,
) -> dict[str, object]:
    return {
        "chunk_id": int(row["chunk_id"]),
        "file_id": int(row["file_id"]),
        "logical_index": logical_index,
        "source_type": str(row["source_type"]),
        "course": row["course"] or "",
        "file_path": str(row["file_path"]),
        "embedding_model": model_name,
        "embedding_dim": dimension,
    }


def _eligible_chunk_count(
    connection: sqlite3.Connection,
    source_types: Sequence[str],
) -> int:
    unique_source_types = tuple(dict.fromkeys(source_types))
    if not unique_source_types:
        return 0
    where_sql = current_chunk_where_sql(
        unique_source_types, require_non_empty_text=True
    )
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        WHERE {where_sql}
        """,
        unique_source_types,
    ).fetchone()
    return int(row[0])


def _embeddings_total(
    connection: sqlite3.Connection,
    *,
    model_name: str,
    physical_names: Sequence[str],
) -> int:
    if not physical_names:
        return 0
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM embeddings
        WHERE embedding_model = ?
          AND vector_collection IN ({placeholders(physical_names)})
        """,
        (model_name, *physical_names),
    ).fetchone()
    return int(row[0])


def _query_candidates(
    client: object,
    *,
    selected: Sequence[tuple[str, str]],
    profile: EmbeddingProfile,
    dimension: int,
    query_vector: Sequence[float],
    limit: int,
) -> dict[int, _Candidate]:
    existing = _existing_collection_names(client)
    candidates: dict[int, _Candidate] = {}
    for logical_index, _source_type in selected:
        physical = _physical_name(logical_index, profile, dimension)
        if physical not in existing:
            continue
        collection = client.get_collection(name=physical)  # type: ignore[attr-defined]
        count = collection.count()
        if count == 0:
            continue
        fetch_k = min(count, max(limit * 4, limit))
        result = collection.query(
            query_embeddings=[list(query_vector)],
            n_results=fetch_k,
            include=["distances", "metadatas"],
        )
        for chunk_id, distance, vector_id in _iter_query_hits(result):
            current = candidates.get(chunk_id)
            if current is None or distance < current.distance:
                candidates[chunk_id] = _Candidate(
                    chunk_id=chunk_id,
                    distance=distance,
                    physical_collection=physical,
                    vector_id=vector_id,
                )
    return candidates


def _iter_query_hits(result: object) -> Iterator[tuple[int, float, str]]:
    ids = _first_or_empty(result, "ids")
    distances = _first_or_empty(result, "distances")
    metadatas = _first_or_empty(result, "metadatas")
    for vector_id, distance, metadata in zip(ids, distances, metadatas):
        md = metadata or {}
        chunk_id = md.get("chunk_id")
        if chunk_id is None and isinstance(vector_id, str) and ":" in vector_id:
            chunk_id = vector_id.split(":", 1)[1]
        if chunk_id is None:
            continue
        try:
            yield int(chunk_id), float(distance), str(vector_id)
        except (TypeError, ValueError):
            continue


def _first_or_empty(result: object, key: str) -> list[object]:
    value = result.get(key) if isinstance(result, dict) else None
    if not value:
        return []
    first = value[0]
    return list(first) if first else []


def _hydrate_candidates(
    config: Config,
    *,
    candidate_ids: Sequence[int],
    source_types: Sequence[str],
    course: str | None,
) -> list[sqlite3.Row]:
    unique_source_types = tuple(dict.fromkeys(source_types))
    if not unique_source_types or not candidate_ids:
        return []
    where_sql = current_chunk_where_sql(
        unique_source_types, require_non_empty_text=False
    )
    sql = f"""
        SELECT
            chunks.id AS chunk_id,
            files.id AS file_id,
            courses.name AS course,
            files.path AS file_path,
            chunks.source_type AS source_type,
            chunks.location_type AS location_type,
            chunks.location_value AS location_value,
            chunks.text AS text
        FROM chunks
        JOIN files ON files.id = chunks.file_id
        LEFT JOIN courses ON courses.id = files.course_id
        WHERE {where_sql}
          AND chunks.id IN ({placeholders(candidate_ids)})
    """
    params: list[object] = [*unique_source_types, *candidate_ids]
    if course is not None:
        sql += " AND LOWER(courses.name) = LOWER(?)"
        params.append(course)
    with closing(connect_sqlite_read_only(config)) as connection:
        return connection.execute(sql, params).fetchall()


def _sync_diagnostics(
    *,
    chunks_seen: int,
    vectors_indexed: int,
    embeddings_total: int,
    model_name: str,
) -> list[str]:
    diagnostics: list[str] = []
    if chunks_seen == 0:
        diagnostics.append("No eligible indexed chunks found for vector indexing.")
    elif vectors_indexed == 0:
        diagnostics.append(
            f"No new chunks to embed; {embeddings_total} eligible chunk(s) "
            f"already embedded for model '{model_name}'."
        )
    return diagnostics


def _chroma_client(config: Config, *, error: type[Exception]) -> object:
    chromadb = _require_chromadb(error=error)
    from chromadb.config import Settings

    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _require_chromadb(*, error: type[Exception]) -> object:
    try:
        import chromadb
    except ImportError as exc:
        raise error(
            "ChromaDB is required for vector features. Install dependencies with: "
            "uv sync"
        ) from exc
    return chromadb


def _existing_collection_names(client: object) -> set[str]:
    names: set[str] = set()
    for collection in client.list_collections():  # type: ignore[attr-defined]
        names.add(collection if isinstance(collection, str) else collection.name)
    return names


def _snippet(text: object) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if not normalized:
        return ""
    if len(normalized) <= _SNIPPET_CHAR_LIMIT:
        return normalized
    return normalized[: _SNIPPET_CHAR_LIMIT - 3].rstrip() + "..."


def _batches(rows: Sequence[sqlite3.Row], size: int) -> Iterable[list[sqlite3.Row]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
