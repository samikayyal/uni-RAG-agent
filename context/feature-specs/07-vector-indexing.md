# Feature Spec 07: Vector Indexing

## Purpose

Embed extracted chunks into ChromaDB collections using LangChain embedding abstractions, while keeping SQLite authoritative for metadata and chunk text.

## Depends On

- [02-configuration-and-storage.md](02-configuration-and-storage.md)
- [04-text-extraction-and-chunking.md](04-text-extraction-and-chunking.md)
- [05-data-schema-summaries.md](05-data-schema-summaries.md)
- `context/architecture.md` tables: `chunks`, `embeddings`, `files`
- DEC-010, DEC-011, DEC-012

## In Scope

- Configure embeddings through LangChain.
- Provide deterministic fake embeddings for tests.
- Create one ChromaDB collection per logical index.
- Embed eligible chunks.
- Store Chroma vector IDs and embedding metadata in SQLite.
- Implement semantic search over selected collections.
- Keep metadata-only files out of vector indexing.
- Add or update the stage EDA notebook for vector-index output once this feature is implemented.

## Out of Scope

- Choosing a required paid/cloud provider.
- Reranking.
- Query routing.
- Embedding source files directly.
- Embedding images, media, archives, binaries, installers, or model artifacts.

## Public Interfaces

Commands:

```powershell
uv run -m uni_rag_agent index vector
uv run -m uni_rag_agent index vector --collection document_index
uv run -m uni_rag_agent index vector --model BAAI/bge-m3
uv run -m uni_rag_agent index vector --model BAAI/bge-m3 --rebuild
uv run -m uni_rag_agent search semantic "distributed computation"
uv run -m uni_rag_agent search semantic "distributed computation" --model BAAI/bge-m3 --index slides_index --course "Information Retrieval" --top-k 10 --json
```

Notebook:

```text
notebooks/vector_index_eda.ipynb
```

Create this notebook when vector indexing is implemented. It should inspect `embeddings`, joined chunk/file/course metadata, Chroma collection metadata, model/dimension consistency, collection sizes, missing embeddings, and small semantic query smoke results.

Internal interfaces:

```python
get_embedding_model(config: Config, model: str | None = None) -> Embeddings
sync_vector_index(
    config: Config,
    collection: str | None = None,
    model: str | None = None,
    rebuild: bool = False,
) -> VectorIndexResult
semantic_search(
    config: Config,
    query: str,
    course: str | None = None,
    indexes: Sequence[str] | None = None,
    top_k: int | None = None,
    model: str | None = None,
) -> list[RetrievalResult]
```

Signature reconciliation: `semantic_search(config, query, ...)` intentionally
mirrors the implemented `keyword_search(config, query, ...)` rather than the
earlier query-first sketch, so the two direct-search entry points share one
shape (config first, query second, then `course`/`indexes`/`top_k`). `top_k`
defaults to `UNI_RAG_SEMANTIC_TOP_K` when omitted.

### Embedding model selection

- Without `--model`, selection follows config: when `UNI_RAG_USE_FAKE_EMBEDDINGS`
  is true `get_embedding_model` returns the deterministic fake adapter; when it
  is false the configured `UNI_RAG_EMBEDDING_MODEL` must resolve to a known real
  profile or the command fails clearly.
- An explicit real `--model BAAI/bge-m3` overrides `UNI_RAG_USE_FAKE_EMBEDDINGS=true`
  for that command so experiments do not require editing `.env`.
- An explicit `--model fake-embedding` is always allowed for fake runs.

### Dependencies

Core dependencies are `chromadb` and `langchain-core`. Real Hugging Face local
models live in the optional `embeddings` extra (`langchain-huggingface` plus the
Sentence Transformers stack, which pulls in `transformers` and `torch`). Those
imports are lazy and only happen when a real Hugging Face profile is selected,
so the fake-default test path stays offline and lightweight. Install the extra
with `uv sync --extra embeddings`.

### Side-by-side models and collections

The public logical collections stay stable (`document_index`, `slides_index`,
etc.). Each embedding model/profile persists into a distinct physical ChromaDB
collection named `<logical_index>__<model_slug>__<hash>`, where the hash input
includes provider, model, dimension, and metric. Collections use cosine
distance. `embeddings` rows store `vector_backend='chroma'`, the physical
`vector_collection`, a stable `vector_id='chunk:<chunk_id>'`, the selected
`embedding_model`, the dimension, and a timestamp. The physical collection is
the canonical profile identity: a chunk may have one mapping per physical
collection, so fake and real runs or dimension rollovers cannot suppress one
another. Fake embeddings always use the stable `fake-embedding` identity even
when config names a real model.

### Shared retrieval contract

`RetrievalResult` (shared with keyword search) keeps `snippet` required and adds
nullable `vector_collection` and `vector_id`. Keyword results emit `null` for
both. Semantic results hydrate `snippet` from a truncated SQLite chunk-text
preview and populate `vector_collection`/`vector_id`. Empty semantic results
return `[]`; "clear diagnostics" means the CLI message plus JSONL telemetry, not
a new return-value field.

Chroma collections:

```text
document_index
slides_index
notebook_index
code_index
data_schema_index
transcript_index
```

Source types map to Chroma collections as follows:

```text
document -> document_index
slides -> slides_index
notebook -> notebook_index
code -> code_index
data_schema -> data_schema_index
transcript -> transcript_index
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
retrieval_method=semantic
vector_collection
vector_id
```

## Storage and Schema Impact

Write ChromaDB files under:

```text
data/indexes/vector/
```

Populate:

- `embeddings`

Read:

- `chunks`
- `files`
- `courses`

SQLite remains authoritative. Chroma metadata should include chunk ID and enough filter fields for efficient search, but chunk text and citations should be loaded from SQLite. A semantic hit is valid only when its exact backend, physical collection, vector ID, and chunk mapping still exist in `embeddings`; then SQLite may hydrate its metadata and text.

## Workflow

1. Load config and embedding adapter.
2. Map chunks to logical Chroma collections by `source_type`.
3. Reconcile each selected physical collection with SQLite: delete Chroma-only vectors and stale SQLite mappings, and make SQLite mappings with missing Chroma vectors eligible for re-embedding.
4. Select chunks missing embeddings for the configured physical profile.
5. Batch embed chunk text.
6. Upsert vectors into ChromaDB.
7. Store `embeddings` rows with vector backend, collection, vector ID, model, dimension, and timestamp.
8. Implement semantic search by querying selected collections, validating exact `embeddings` mappings, and then joining result IDs back to SQLite metadata. Apply course filtering before final top-K truncation.
8. Keep `notebooks/vector_index_eda.ipynb` aligned with embedding fields, collection names, vector metadata, model/dimension semantics, and search result shape.

## Failure and Safety Rules

- Tests must not require network or API keys.
- If provider credentials are missing and fake embeddings are disabled, fail with a clear config error.
- Do not embed chunks from metadata-only files.
- Do not embed empty or whitespace-only chunks.
- Batch failures should be recoverable without corrupting existing embeddings.
- Incremental sync must repair a missing selected Chroma collection/vector and remove Chroma vectors that no longer have current authoritative mappings.
- Semantic search must ignore a stale Chroma vector even if SQLite later reuses its numeric chunk ID for a different chunk.
- Do not store secrets in Chroma metadata.
- The EDA notebook must read generated app data only and must not mutate SQLite, ChromaDB files, or `Courses`.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated tests with deterministic fake embeddings.
- Verify collection mapping for document, slide, notebook, code, data schema, and transcript chunks.
- Verify embedding sync is idempotent.
- Verify fake-to-real and embedding-dimension rollovers create distinct physical mappings and remain searchable.
- Verify incremental sync removes orphaned vectors and restores missing collections/vectors.
- Verify semantic search returns chunk metadata joined from SQLite.
- Verify semantic search validates the exact SQLite mapping and applies course filters before final top-K truncation.
- Verify metadata-only files are never embedded.
- Verify missing real provider config fails clearly when fake embeddings are disabled.
- Verify `notebooks/vector_index_eda.ipynb`, once created, is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Optional smoke: vector-index a tiny fixture corpus with fake embeddings.

## Acceptance Criteria

- `uv run -m uni_rag_agent index vector` populates ChromaDB and `embeddings`.
- Semantic search works across selected logical collections.
- The implementation is provider-configurable and testable without API keys.
- SQLite remains the source of truth for chunk text, paths, and citation metadata.
- `notebooks/vector_index_eda.ipynb` exists once this feature lands and can inspect embedding/vector coverage without mutating generated or source data.
