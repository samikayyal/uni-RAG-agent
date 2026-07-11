"""Read-only inventory metadata retrieval for file and course discovery."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing

from rapidfuzz import fuzz

from uni_rag_agent.config import Config
from uni_rag_agent.indexing.eligibility import INDEX_TO_SOURCE_TYPE
from uni_rag_agent.storage import connect_sqlite_read_only

from .models import RetrievalResult

MISSING_INVENTORY_REASON = "missing from latest inventory run"
METADATA_ONLY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".jfif",
    ".zip",
    ".rar",
    ".7z",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4a",
    ".wav",
    ".exe",
    ".msi",
    ".cab",
    ".bin",
    ".joblib",
    ".weights",
    ".tflite",
    ".pt",
    ".pkl",
    ".rdata",
    ".rds",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "course",
    "courses",
    "did",
    "do",
    "explain",
    "find",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "locate",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "the",
    "to",
    "what",
    "where",
    "with",
    "about",
    "does",
    "this",
    "that",
    "was",
    "were",
    "you",
    "your",
}


class MetadataSearchError(ValueError):
    """Raised when a metadata query cannot be executed safely."""


def metadata_search(
    config: Config,
    query: str,
    *,
    courses: Sequence[str] | None = None,
    indexes: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    limit = top_k if top_k is not None else config.metadata_top_k
    if limit <= 0:
        raise MetadataSearchError("top_k must be greater than zero")
    normalized_query = " ".join(query.strip().split())
    if not normalized_query:
        raise MetadataSearchError("Metadata query must not be empty.")
    if courses is not None and not courses:
        return []
    normalized_extensions = tuple(
        extension.casefold()
        if extension.startswith(".")
        else f".{extension.casefold()}"
        for extension in (extensions or ())
    )
    allowed_categories = _allowed_categories(indexes)
    requested_courses = tuple(courses or ())
    terms = _query_terms(normalized_query, requested_courses)
    if not terms and not normalized_extensions:
        return []

    try:
        with closing(connect_sqlite_read_only(config)) as connection:
            rows = _load_file_rows(
                connection,
                courses=requested_courses if courses is not None else None,
                allowed_categories=allowed_categories,
                extensions=normalized_extensions,
            )
    except sqlite3.Error as exc:
        raise MetadataSearchError(
            f"Metadata search could not inspect SQLite: {exc}"
        ) from exc

    ranked: list[tuple[tuple[object, ...], sqlite3.Row, tuple[str, ...], float]] = []
    for row in rows:
        match = _score_file(
            row,
            query=normalized_query,
            terms=terms,
            extensions=normalized_extensions,
            filename_threshold=config.filename_fuzzy_threshold,
            path_threshold=config.path_fuzzy_threshold,
        )
        if match is None:
            continue
        tier, native_score, matched_fields = match
        ranked.append(
            (
                (
                    tier,
                    -native_score,
                    len(str(row["path"])),
                    str(row["path"]).casefold(),
                    int(row["id"]),
                ),
                row,
                matched_fields,
                native_score,
            )
        )
    ranked.sort(key=lambda item: item[0])

    results: list[RetrievalResult] = []
    for rank, (_sort_key, row, matched_fields, native_score) in enumerate(
        ranked[:limit], start=1
    ):
        reason = row["reason_not_indexed"]
        snippet = " | ".join(
            value
            for value in (
                str(row["filename"]),
                str(row["category"]),
                str(row["index_status"]),
                str(reason) if reason else "",
            )
            if value
        )
        results.append(
            RetrievalResult(
                chunk_id=None,
                file_id=int(row["id"]),
                course=row["course"],
                file_path=str(row["path"]),
                source_type=None,
                location_type=None,
                location_value=None,
                rank=rank,
                score=native_score,
                snippet=snippet,
                retrieval_method="metadata",
                file_category=row["category"],
                file_index_status=row["index_status"],
                reason_not_indexed=reason,
                matched_fields=matched_fields,
            )
        )
    return results


def _allowed_categories(indexes: Sequence[str] | None) -> tuple[str, ...] | None:
    if indexes is None:
        return None
    if not indexes:
        return ()
    categories: list[str] = []
    for index in indexes:
        source_type = INDEX_TO_SOURCE_TYPE.get(index)
        if source_type is not None and source_type not in categories:
            categories.append(source_type)
    unknown = [index for index in indexes if index not in INDEX_TO_SOURCE_TYPE]
    if unknown:
        raise MetadataSearchError(
            f"Unknown logical index name(s): {', '.join(unknown)}"
        )
    return tuple(categories)


def _load_file_rows(
    connection: sqlite3.Connection,
    *,
    courses: Sequence[str] | None,
    allowed_categories: Sequence[str] | None,
    extensions: Sequence[str],
) -> list[sqlite3.Row]:
    where = [
        "COALESCE(files.reason_not_indexed, '') <> ?",
    ]
    params: list[object] = [MISSING_INVENTORY_REASON]
    if courses is not None:
        where.append(f"LOWER(courses.name) IN ({', '.join('?' for _ in courses)})")
        params.extend(course.casefold() for course in courses)
    if allowed_categories is not None:
        if not allowed_categories:
            # A metadata-only extension can still be found when it was named
            # explicitly, but no extractable category is otherwise allowed.
            if not any(
                extension in METADATA_ONLY_EXTENSIONS for extension in extensions
            ):
                return []
            where.append(f"files.extension IN ({', '.join('?' for _ in extensions)})")
            params.extend(extensions)
        else:
            category_clause = (
                f"files.category IN ({', '.join('?' for _ in allowed_categories)})"
            )
            params.extend(allowed_categories)
            metadata_extension_clause = (
                "files.extension IN (" + ", ".join("?" for _ in extensions) + ")"
                if extensions
                else "0"
            )
            params.extend(extensions)
            where.append(f"({category_clause} OR {metadata_extension_clause})")
    sql = f"""
        SELECT
            files.id,
            files.path,
            files.relative_path,
            files.filename,
            files.extension,
            files.category,
            files.index_status,
            files.reason_not_indexed,
            courses.name AS course
        FROM files
        LEFT JOIN courses ON courses.id = files.course_id
        WHERE {" AND ".join(where)}
    """
    return connection.execute(sql, params).fetchall()


def _query_terms(query: str, courses: Sequence[str]) -> tuple[str, ...]:
    course_tokens = {
        token.casefold() for course in courses for token in re.findall(r"\w+", course)
    }
    terms: list[str] = []
    seen: set[str] = set()
    for quoted in re.findall(r'"([^"\n]+)"', query):
        text = quoted.strip()
        if text and text.casefold() not in seen:
            terms.append(text)
            seen.add(text.casefold())
    without_quotes = re.sub(r'"[^"\n]+"', " ", query)
    for raw in re.findall(r"[^\s]+", without_quotes):
        token = raw.strip(".,;:!?()[]{}")
        if (
            not token
            or token.casefold() in STOPWORDS
            or token.casefold() in course_tokens
        ):
            continue
        if token.casefold() not in seen:
            terms.append(token)
            seen.add(token.casefold())
    return tuple(terms)


def _score_file(
    row: sqlite3.Row,
    *,
    query: str,
    terms: Sequence[str],
    extensions: Sequence[str],
    filename_threshold: int,
    path_threshold: int,
) -> tuple[int, float, tuple[str, ...]] | None:
    filename = str(row["filename"])
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    path = str(row["path"])
    relative_path = str(row["relative_path"])
    extension = str(row["extension"]).casefold()
    category = str(row["category"]).casefold()
    query_key = query.casefold()
    filename_key = filename.casefold()
    stem_key = stem.casefold()
    path_key = path.casefold()
    relative_key = relative_path.casefold()
    matched: list[str] = []
    best_fuzzy = 0.0

    if query_key in {filename_key, path_key, relative_key} or any(
        term.casefold() in {filename_key, path_key, relative_key} for term in terms
    ):
        if query_key == filename_key or any(
            term.casefold() == filename_key for term in terms
        ):
            matched.append("filename")
        else:
            matched.append("path")
        return 1, 100.0, tuple(dict.fromkeys(matched))
    if any(term.casefold() == stem_key for term in terms):
        return 2, 100.0, ("filename_stem",)
    if any(
        stem_key.startswith(term.casefold()) or filename_key.startswith(term.casefold())
        for term in terms
    ):
        return 3, 95.0, ("filename_prefix",)
    if extensions and extension in extensions:
        matched.append("extension")
    if any(_category_matches(category, term) for term in terms):
        matched.append("category")
    if matched:
        return 4, 90.0, tuple(dict.fromkeys(matched))
    if any(term.casefold() in filename_key for term in terms):
        return 5, 80.0, ("filename",)
    if any(term.casefold() in relative_key for term in terms):
        return 6, 70.0, ("path",)
    for term in terms:
        best_fuzzy = max(
            best_fuzzy,
            float(fuzz.WRatio(term, filename)),
            float(fuzz.WRatio(term, stem)),
        )
    if best_fuzzy >= 0:
        # Thresholds are applied by the caller's configuration in the public
        # function; use the row-level score here and filter in _score_threshold.
        pass
    return _score_threshold(
        row,
        terms=terms,
        score=best_fuzzy,
        filename_threshold=filename_threshold,
        path_threshold=path_threshold,
    )


def _score_threshold(
    row: sqlite3.Row,
    *,
    terms: Sequence[str],
    score: float,
    filename_threshold: int,
    path_threshold: int,
) -> tuple[int, float, tuple[str, ...]] | None:
    # This helper is patched by the public wrapper below so filename and path
    # thresholds remain independent without reading source files.
    if score >= filename_threshold:
        return 7, score, ("filename_fuzzy",)
    path_score = max(
        (float(fuzz.WRatio(term, str(row["relative_path"]))) for term in terms),
        default=0.0,
    )
    if path_score >= path_threshold:
        return 8, path_score, ("path_fuzzy",)
    return None


def _category_matches(category: str, term: str) -> bool:
    value = term.casefold()
    return value in {category, category.replace("_", " ")}
