# Feature Spec 07: Vector Indexing

## Purpose

Embed extracted chunks into ChromaDB collections using provider-inferred embedding abstractions, while keeping SQLite authoritative for metadata and chunk text. The profile registry supports reviewed local Hugging Face execution plus reviewed hosted Google Gemini and Nebius Token Factory construction.

## Depends On

- [02-configuration-and-storage.md](02-configuration-and-storage.md)
- [04-text-extraction-and-chunking.md](04-text-extraction-and-chunking.md)
- [05-data-schema-summaries.md](05-data-schema-summaries.md)
- `context/architecture.md` tables: `chunks`, `embeddings`, `files`
- DEC-010, DEC-011, DEC-012, DEC-031, DEC-039

## In Scope

- Configure embeddings through LangChain.
- Restrict production selection to the explicit reviewed embedding-profile registry.
- Infer provider construction from the selected registry profile; do not add a
  generic plugin framework or `UNI_RAG_EMBEDDING_PROVIDER`.
- Inject deterministic test-only embeddings at the model-loader boundary.
- Create one ChromaDB collection per logical index.
- Embed eligible chunks.
- Store Chroma vector IDs and embedding metadata in SQLite.
- Implement semantic search over selected collections.
- Keep metadata-only files out of vector indexing.
- Load local and hosted SDKs lazily through provider-specific optional extras.
- Apply shared vector validation/retry rules with 64-chunk batches, per-batch
  commits, and incremental resumability after exhausted hosted retries.
- Add or update the stage EDA notebook for vector-index output once this feature is implemented.

## Out of Scope

- Adding unreviewed providers or profiles, or inventing a generic provider/plugin
  framework.
- Vertex AI construction.
- Reranking.
- Query routing.
- Embedding source files directly.
- Embedding images, media, archives, binaries, installers, or model artifacts.

## Public Interfaces

Commands:

```powershell
uv run -m uni_rag_agent index vector --model BAAI/bge-m3
uv run -m uni_rag_agent index vector --model BAAI/bge-m3 --collection document_index
uv run -m uni_rag_agent index vector --model BAAI/bge-m3 --rebuild
uv run -m uni_rag_agent search semantic "distributed computation" --model BAAI/bge-m3
uv run -m uni_rag_agent search semantic "distributed computation" --model BAAI/bge-m3 --index slides_index --course "Information Retrieval" --top-k 10 --json
uv run -m uni_rag_agent index vector --model gemini-embedding-001 --rebuild
uv run -m uni_rag_agent search semantic "distributed computation" --model gemini-embedding-001 --json
uv run -m uni_rag_agent index vector --model Qwen/Qwen3-Embedding-8B --rebuild
uv run -m uni_rag_agent search semantic "distributed computation" --model Qwen/Qwen3-Embedding-8B --json
```

Notebook:

```text
notebooks/vector_index_eda.ipynb
```

Create this notebook when vector indexing is implemented. It should inspect
`embeddings`, joined chunk/file/course metadata, Chroma collection metadata,
provider-neutral canonical profile identity, declared/runtime dimension
consistency, collection sizes, missing embeddings, and small semantic query
smoke results. It must not group every profile as Hugging Face: the provider is
inferred from the registry and hosted profiles are a first-class interpretation.

Internal interfaces:

```python
build_embedding_model(
    config: Config,
    model: str | None = None,
) -> BuiltEmbeddingModel
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
    *,
    courses: Sequence[str] | None = None,
) -> list[RetrievalResult]
```

Signature reconciliation: `semantic_search(config, query, ...)` intentionally
mirrors the implemented `keyword_search(config, query, ...)` rather than the
earlier query-first sketch, so the two direct-search entry points share one
shape (config first, query second, then `course`/`indexes`/`top_k`). `top_k`
defaults to `UNI_RAG_SEMANTIC_TOP_K` when omitted.

`courses` is the plural Feature 08 compatibility extension. It is resolved to
canonical SQLite course spelling, filtered in Chroma before top-K, and reapplied
during authoritative SQLite hydration. The singular `course` remains supported;
passing both is an error and an empty plural sequence returns no results.

### Embedding model selection and reviewed profiles

- Resolve a nonblank explicit `--model` first, then nonblank
  `UNI_RAG_EMBEDDING_MODEL`/`config.embedding_model`.
- If neither value is set, raise the caller's domain-specific error and explain
  how to set `UNI_RAG_EMBEDDING_MODEL` or pass `--model`, including the complete
  supported profile list.
- Resolve aliases through the registry before construction. The accepted alias
  `gemini-embedding-001` canonicalizes to `google/gemini-embedding-001`.
- Infer the provider from the registry. There is no
  `UNI_RAG_EMBEDDING_PROVIDER`, and no caller-supplied provider override.
- Accept exactly these reviewed profiles and declared dimensions:

| Registry identifier | Provider / execution | Declared dimension | Optional extra |
| :--- | :--- | :---: | :--- |
| `BAAI/bge-m3` | Hugging Face / local | 1024 | `embeddings` |
| `jinaai/jina-embeddings-v3` | Hugging Face / local | 1024 | `embeddings` |
| `jinaai/jina-embeddings-v5-text-small` | Hugging Face / local | 1024 | `embeddings` |
| `google/embeddinggemma-300m` | Hugging Face / local | 768 | `embeddings` |
| `google/gemini-embedding-001` (`gemini-embedding-001` alias) | Google Gemini API / hosted | 3072 | `embeddings-cloud` |
| `Qwen/Qwen3-Embedding-8B` | Nebius Token Factory / hosted | 4096 | `embeddings-cloud` |

- Reject unknown identifiers generically with the supported profile list. Do not
  provide an offline production selection.

### Dependencies and provider construction

Core dependencies are `chromadb` and `langchain-core`. Local Hugging Face models
live in the optional `embeddings` extra (`langchain-huggingface` plus the Sentence
Transformers stack, which pulls in `transformers` and `torch`). Hosted Google and
Nebius construction lives in the optional `embeddings-cloud` extra. Install only
the path required by the selected profile:

```powershell
# Local Hugging Face.
uv sync --extra embeddings

# Hosted Google Gemini or Nebius Token Factory.
uv sync --extra embeddings-cloud
```

The existing LLM integration extra remains separate and unchanged:
`uv sync --extra llm`. All embedding SDK imports and client construction are
lazy and occur only after a reviewed profile is selected. Automated tests inject
a deterministic LangChain embedding object at the loader boundary and continue
to use real ChromaDB and SQLite.

Hosted construction is explicit and has no Vertex AI path. The Google profile
uses the direct Gemini API with `GOOGLE_API_KEY`. The Nebius profile uses the
fixed base URL `https://api.tokenfactory.nebius.com/v1/`, `NEBIUS_API_KEY`, and
the exact model `Qwen/Qwen3-Embedding-8B`. Query inputs for that profile use:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{query}
```

The instruction is for semantic queries; document/chunk text is sent as the
document input. Missing optional extras and missing credentials produce
sanitized setup diagnostics without exposing key values or provider response
payloads.

### Side-by-side models and collections

The public logical collections stay stable (`document_index`, `slides_index`,
etc.). Each embedding profile persists into a distinct physical ChromaDB
collection named `<logical_index>__<model_slug>__<hash>`, where the hash input
includes provider, canonical model, dimension, and metric. Collections use
cosine distance. `embeddings` rows store `vector_backend='chroma'`, the physical
`vector_collection`, a stable `vector_id='chunk:<chunk_id>'`, the canonical
`embedding_model`, the validated dimension, and a timestamp. The physical
collection is the canonical storage identity: a chunk may have one mapping per
physical collection, so side-by-side reviewed profiles cannot suppress one
another.

Local Hugging Face construction probes the runtime dimension after loading the
model. Hosted construction uses the declared profile dimension and makes no
dedicated probe call; each actual hosted response is validated against the
declared dimension during its normal batch. The validated dimension drives
collection identity and SQLite telemetry. The canonical model identity is also
carried into `RetrievalRun`, persisted retrieval/evidence settings, and vector,
retrieval, and evidence telemetry; an accepted alias is never persisted as a
second identity.

Shared response validation requires the expected vector count, nonempty finite
numeric vectors, and the expected dimension. Shared retries allow three total
attempts for network failures, HTTP 408/429, and HTTP 5xx responses. Malformed or
dimension-invalid responses, authentication/permission failures, model failures,
and other HTTP 4xx responses fail immediately without retry. Provider response
ordering is validated where an API supplies indexes, but returned order is not
silently reordered.

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

1. Resolve the explicit/configured reviewed model, canonicalize any alias, infer
   its provider from the registry, and load the provider lazily.
2. Map chunks to logical Chroma collections by `source_type`.
3. Reconcile each selected physical collection with SQLite: delete Chroma-only vectors and stale SQLite mappings, and make SQLite mappings with missing Chroma vectors eligible for re-embedding.
4. Select chunks missing embeddings for the configured physical profile.
5. Embed eligible chunk text in batches of exactly 64 or fewer for the final
   batch. Apply the same vector-shape validation and bounded retry policy to
   local and hosted calls; a batch must return one valid vector per input and the
   expected dimension before it is accepted.
6. Upsert the validated batch into ChromaDB and commit its SQLite mapping rows in
   the same per-batch unit. Do not commit a partially validated batch.
7. Store `embeddings` rows with vector backend, collection, vector ID, canonical
   model, validated dimension, and timestamp.
8. If a hosted batch exhausts its retries, fail with a sanitized diagnostic after
   retaining earlier commits. A later incremental run must discover and retry
   only the missing chunks, so a transient hosted failure does not require a
   complete rebuild.
9. Implement semantic search by querying selected collections, validating exact
   `embeddings` mappings, and then joining result IDs back to SQLite metadata.
   Apply course filtering before final top-K truncation.
10. Keep `notebooks/vector_index_eda.ipynb` aligned with embedding fields,
    provider-neutral canonical profile identity, collection names, declared vs
    observed dimension semantics, and search result shape.

## Failure and Safety Rules

- Tests must not require network or API keys; they inject test-only model doubles.
- If the optional embedding dependency or model access is unavailable, fail with
  a clear sanitized installation/access diagnostic. Local failures name
  `uv sync --extra embeddings`; hosted failures name
  `uv sync --extra embeddings-cloud` or the required credential variable without
  exposing its value.
- Provider construction is inferred from the registry. Do not accept or document
  `UNI_RAG_EMBEDDING_PROVIDER`, Vertex AI settings, or a generic provider/plugin
  override.
- Local runtime dimensions must be probed and validated. Hosted profiles must
  validate every actual returned vector against the declared dimension and must
  not issue a dedicated hosted probe request.
- Use the shared validation/retry policy for all providers, process at most 64
  chunks per batch, and commit each successful batch separately. Exhausted
  hosted retries retain earlier commits and leave incremental sync resumable.
- Do not embed chunks from metadata-only files.
- Do not embed empty or whitespace-only chunks.
- Batch failures should be recoverable without corrupting existing embeddings.
- Incremental sync must repair a missing selected Chroma collection/vector and remove Chroma vectors that no longer have current authoritative mappings.
- Semantic search must ignore a stale Chroma vector even if SQLite later reuses its numeric chunk ID for a different chunk.
- Do not store secrets in Chroma metadata.
- The EDA notebook must read generated app data only and must not mutate SQLite, ChromaDB files, or `Courses`.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated tests with `DeterministicTestEmbeddings` injected at the provider
  constructor boundary, plus a subprocess-only constructor shim for CLI coverage.
- Verify collection mapping for document, slide, notebook, code, data schema, and transcript chunks.
- Verify embedding sync is idempotent.
- Verify local and hosted registry profiles (with injected dimensions) create
  distinct physical mappings, preserve canonical identity/alias behavior, and
  remain searchable.
- Verify declared hosted dimensions are checked against actual vectors without a
  dedicated hosted probe, and verify the shared validation/retry boundary.
- Verify 64-chunk batching, per-batch commits, and incremental resumption after
  an exhausted hosted retry.
- Verify incremental sync removes orphaned vectors and restores missing collections/vectors.
- Verify semantic search returns chunk metadata joined from SQLite.
- Verify semantic search validates the exact SQLite mapping and applies course filters before final top-K truncation.
- Verify metadata-only files are never embedded.
- Verify missing model selection fails clearly for indexing and semantic search.
- Verify `notebooks/vector_index_eda.ipynb`, once created, is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Manual credentialed smokes are optional: after `uv sync --extra embeddings`,
  select a local reviewed model and index a tiny corpus; after
  `uv sync --extra embeddings-cloud`, optionally smoke the Gemini or Nebius
  profile with its credential. Do not make network/model loading part of pytest.

## Acceptance Criteria

- `uv run -m uni_rag_agent index vector --model <reviewed-profile>` populates ChromaDB and `embeddings`.
- Semantic search works across selected logical collections.
- The implementation is configured from reviewed profiles, infers providers from
  the registry, and is testable without API keys through injected doubles.
- SQLite remains the source of truth for chunk text, paths, and citation metadata.
- `notebooks/vector_index_eda.ipynb` exists once this feature lands and can inspect embedding/vector coverage without mutating generated or source data.
