# Keyword and vector indexing

## Current behavior

`search_contracts.py` owns the one logical-index/source-type taxonomy. The
`indexing/eligibility.py` helpers reuse its derived eligible source types and
inverse lookups, so only chunks joined to `files.index_status = 'indexed'` and
one of six eligible source types are indexed. `sync_keyword_index()` rebuilds
SQLite FTS5 `chunk_fts` (unicode61) from that set, applying NFKC normalization
to chunk text and titles in Python (SQLite has no NFKC function) so
compatibility forms — notably Arabic Presentation Forms emitted by PDF
extraction — match normally-typed queries. Query terms are NFKC-normalized
symmetrically at search time. `keyword_search()` supports plain-text terms,
exact course filters, logical-index filters, and bounded results; direct
search is read-only. Planner-supplied keyword phrases are NFKC-normalized,
tokenized, deduplicated, and joined with OR for recall instead of being forced
to match as exact multi-word FTS phrases. Individual planner terms with no word
or number tokens are ignored; the search fails only when no usable token remains
across all terms. Search projections return
course-relative file paths (`files.relative_path`), never absolute host paths.

`sync_vector_index()` resolves one reviewed embedding profile, maps source types
to logical Chroma indexes, reconciles stale vectors/mappings, embeds missing
chunks in bounded batches, and records SQLite `embeddings` mappings.
Reconciliation also compares each existing vector's stored course/path filter
metadata against authoritative SQLite and updates drifted entries in place
(no re-embedding), so files reassigned to another course by a later inventory
run become semantically reachable after a normal incremental sync. Collections
are cosine, physical, model-namespaced identities. `semantic_search_many()`
reuses one embedding provider, one Chroma client, and one set of collection
handles for all queries in a retrieval request. Hosted providers batch query
vectors; local Hugging Face profiles reuse the loaded model while preserving
the provider's query-specific operation. Chroma receives all query vectors per
collection in one call, then the application returns one independently ranked
result list per input query. `semantic_search()` is the single-query wrapper.
Both seams validate the exact SQLite mapping, reapply current-file/course/index
filters, and do not persist search runs.

Profiles are `BAAI/bge-m3`, `jinaai/jina-embeddings-v3`,
`jinaai/jina-embeddings-v5-text-small`, `google/embeddinggemma-300m`,
`google/gemini-embedding-001` (alias `gemini-embedding-001`), and
`Qwen/Qwen3-Embedding-8B`. Local profiles use `embeddings`; hosted profiles use
`embeddings-cloud`. Provider is inferred from the canonical profile.

## Public entry points

- `uv run -m uni_rag_agent index keyword [--rebuild]`
- `uv run -m uni_rag_agent search keyword "query" [--course ...] [--index ...] [--json]`
- `uv run -m uni_rag_agent index vector --model <profile> [--collection ...] [--rebuild]`
- `uv run -m uni_rag_agent search semantic "query" --model <profile> [--course ...] [--index ...] [--json]`
- Python: `sync_keyword_index`, `keyword_search`, `sync_vector_index`,
  `semantic_search`, `semantic_search_many`, profile resolution, and provider
  builders.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/search_contracts.py` and
  `src/uni_rag_agent/indexing/{eligibility,keyword,vector,profiles}.py`
  and `indexing/embedding_providers/`.
- Tests: `tests/test_keyword_indexing.py`, `tests/test_hybrid_retrieval.py`,
  `tests/test_vector_indexing.py`, `tests/test_embedding_providers.py`.
- Notebooks: `notebooks/keyword_index_eda.ipynb` and
  `notebooks/vector_index_eda.ipynb` (read-only index EDA).
- SQLite: FTS5 `chunk_fts` and `embeddings`; Chroma persistence under
  `data/indexes/vector/`; index commands emit sanitized JSONL under `data/runs/`.

## Invariants and failure boundaries

- FTS5 and Chroma use the same current-file predicate and logical-index mapping;
  stale historical chunks cannot return as answer evidence.
- No model selection means a clear error. SDK imports are lazy. Local dimensions
  are probed; hosted dimensions are declared and actual vectors are validated.
  Batches contain at most 64 chunks and commit independently.
- Retry only network/408/429/5xx failures (three attempts); malformed,
  dimension-invalid, credential/permission, model, and other 4xx failures are
  not retried. Existing successful hosted batches remain durable.
- Canonical model identity is used in SQLite, Chroma, retrieval/evidence
  settings, and telemetry. The Gemini alias never creates another profile.
- Hosted text and semantic queries leave the machine and may incur charges;
  credentials are never emitted.

Binding decisions: [DEC-011](../decisions.md#dec-011--sqlite-authority-with-chroma-logical-indexes),
[DEC-023/029/028/040](../decisions.md#dec-023029028040--current-file-and-deletion-semantics),
and [DEC-031/039](../decisions.md#dec-031039--explicit-reviewed-embedding-profiles).
