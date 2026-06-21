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
uv run -m uni_rag_agent search semantic "distributed computation"
```

Internal interfaces:

```python
get_embedding_model(config: Config) -> Embeddings
sync_vector_index(config: Config, collection: str | None = None) -> VectorIndexResult
semantic_search(
    query: str,
    course: str | None = None,
    indexes: list[str] | None = None,
    top_k: int = 20,
) -> list[RetrievalResult]
```

Chroma collections:

```text
document_index
slides_index
notebook_index
code_index
data_schema_index
transcript_index
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

SQLite remains authoritative. Chroma metadata should include chunk ID and enough filter fields for efficient search, but chunk text and citations should be loaded from SQLite.

## Workflow

1. Load config and embedding adapter.
2. Map chunks to logical Chroma collections by `source_type`.
3. Select chunks missing embeddings for the configured model.
4. Batch embed chunk text.
5. Upsert vectors into ChromaDB.
6. Store `embeddings` rows with vector backend, collection, vector ID, model, dimension, and timestamp.
7. Implement semantic search by querying selected collections and joining result IDs back to SQLite metadata.

## Failure and Safety Rules

- Tests must not require network or API keys.
- If provider credentials are missing and fake embeddings are disabled, fail with a clear config error.
- Do not embed chunks from metadata-only files.
- Do not embed empty or whitespace-only chunks.
- Batch failures should be recoverable without corrupting existing embeddings.
- Do not store secrets in Chroma metadata.

## Tests

- Automated tests with deterministic fake embeddings.
- Verify collection mapping for document, slide, notebook, code, data schema, and transcript chunks.
- Verify embedding sync is idempotent.
- Verify semantic search returns chunk metadata joined from SQLite.
- Verify metadata-only files are never embedded.
- Verify missing real provider config fails clearly when fake embeddings are disabled.
- Optional smoke: vector-index a tiny fixture corpus with fake embeddings.

## Acceptance Criteria

- `uv run -m uni_rag_agent index vector` populates ChromaDB and `embeddings`.
- Semantic search works across selected logical collections.
- The implementation is provider-configurable and testable without API keys.
- SQLite remains the source of truth for chunk text, paths, and citation metadata.
