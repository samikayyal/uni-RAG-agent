# Feature Spec 06: Keyword Indexing

## Purpose

Build reliable exact-term retrieval over extracted chunks using SQLite FTS5. Keyword search should support course, index, file, and source-type filters and provide results suitable for hybrid retrieval and evidence packet construction.

## Depends On

- [02-configuration-and-storage.md](02-configuration-and-storage.md)
- [04-text-extraction-and-chunking.md](04-text-extraction-and-chunking.md)
- [05-data-schema-summaries.md](05-data-schema-summaries.md)
- `context/architecture.md` tables: `chunks`, `chunk_fts`, `files`, `courses`
- DEC-005, DEC-014

## In Scope

- Create and maintain the `chunk_fts` FTS5 table.
- Index chunk text, title, course name, and file path.
- Rebuild the FTS table from a joined projection of `chunks`, `files`, and `courses`.
- Incrementally sync new/updated chunks when practical.
- Implement keyword search with filters and ranking.
- Return chunk IDs and enough metadata for retrieval merging.

## Out of Scope

- Vector search.
- Query routing.
- Evidence packet assembly.
- Reranking.
- Custom external keyword engines such as Tantivy or Whoosh.

## Public Interfaces

Commands:

```powershell
uv run -m uni_rag_agent index keyword
uv run -m uni_rag_agent index keyword --rebuild
uv run -m uni_rag_agent search keyword "mapreduce"
```

Internal interfaces:

```python
sync_keyword_index(config: Config, rebuild: bool = False) -> KeywordIndexResult
keyword_search(
    query: str,
    course: str | None = None,
    indexes: list[str] | None = None,
    top_k: int = 20,
) -> list[RetrievalResult]
```

Result fields:

```text
chunk_id
file_id
course
file_path
source_type
location_type
location_value
rank
score
snippet
retrieval_method=keyword
```

`source_type` filters use logical chunk categories from `context/architecture.md`, not file extensions. For example, a query scoped to `slides_index` filters chunks with `source_type=slides`; the `.pptx` or `.ppt` extension remains in the joined `files` row.

## Storage and Schema Impact

Maintain:

- `chunk_fts`

Read:

- `chunks`
- `files`
- `courses`

`chunk_fts` is a denormalized FTS projection keyed by `chunk_id`; it is not an external-content FTS table directly bound to `chunks`. The implementation should populate it from `chunks` joined to `files` and `courses` so keyword search can match chunk text, titles, course names, and file paths. It may use rebuild-first behavior for MVP, then add incremental updates later if needed.

## Workflow

1. Validate SQLite FTS5 availability.
2. For rebuilds, clear and repopulate `chunk_fts` from all current chunks joined to their file and course rows.
3. Include chunk text, title, course name, and file path terms in searchable fields.
4. Translate logical index filters to logical chunk source types.
5. Execute FTS query safely.
6. Return top K ranked results with chunk and file metadata.
7. Log keyword query terms and result counts for later search coverage.

## Failure and Safety Rules

- Escape or parameterize user query values to avoid SQL injection.
- Invalid FTS syntax should return a clear query error, not crash the app.
- Empty indexes should return no results with a clear diagnostic.
- Keyword indexing must not read source files under `Courses`; it reads extracted chunks only.
- Keyword search must not mutate source or generated content except optional search logs in later specs.

## Tests

- Automated fixture database with known chunks.
- Verify exact term matches rank above unrelated chunks.
- Verify course and source-type filters work.
- Verify file path and title terms are searchable.
- Verify course name terms are searchable even though course names live in `courses`, not `chunks`.
- Verify invalid query syntax is handled.
- Verify rebuild creates one FTS row per eligible chunk.
- Optional smoke: keyword search over a tiny extracted fixture database.

## Acceptance Criteria

- `uv run -m uni_rag_agent index keyword --rebuild` creates a usable FTS index.
- `keyword_search()` returns stable result objects with chunk/file metadata.
- Search supports exact course and logical-index filters.
- The implementation uses SQLite FTS5, not a separate keyword engine.
