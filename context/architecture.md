# Architecture

This document describes the proposed technical architecture for Uni RAG Agent: storage layout, database schema, indexing model, pipeline stages, tool interfaces, and evidence packet format.

The root `project-overview.md` contains the fuller product narrative. This file is the implementation-facing architecture plan.

## System Goals

The system must:

- inventory every file under `Courses`;
- selectively extract/index useful text-like course knowledge;
- keep images, binaries, archives, installers, unsafe artifacts, and most media metadata-only;
- search through hybrid retrieval;
- build structured evidence packets;
- answer only from retrieved evidence;
- cite files and locations;
- explain weak retrieval.

## Proposed Project Layout

```text
D:\Projects\Uni RAG Agent
|-- Courses\                         # Source archive, read-only from the app's perspective
|-- context\                         # Planning and architecture docs
|   |-- project_overview.md
|   |-- decisions.md
|   |-- progress_tracker.md
|   |-- architecture.md
|   `-- feature-specs\               # MVP module implementation contracts
|-- data\                            # Generated local app data, gitignored
|   |-- uni_rag.sqlite               # Metadata, chunks, search logs, evidence packets
|   |-- extracted\                   # Optional extracted text cache by file hash
|   |-- indexes\
|   |   `-- vector\                  # ChromaDB persistence
|   `-- runs\                        # Optional JSON artifacts for ingestion/search/debug runs
|-- src\
|   `-- uni_rag_agent\
|       |-- __init__.py
|       |-- __main__.py
|       |-- cli.py
|       |-- config.py
|       |-- logging_config.py
|       |-- storage\
|       |-- inventory\
|       |-- extraction\
|       |-- indexing\
|       |-- retrieval\
|       |-- answering\
|       |-- tools\
|       |-- app\
|       `-- evaluation\
|-- tests\
|-- pyproject.toml
|-- .env                             # Configuration and API keys, gitignored
|-- .env.example                     # Template with required env vars, committed
`-- README.md
```

`Courses` should remain source data. Generated files should live under `data\` and should be ignored by git. SQLite FTS5 lives inside `data\uni_rag.sqlite`; do not create a separate keyword index unless a later decision changes the storage design.

## Storage Strategy

Use SQLite as the system of record for:

- file inventory;
- course metadata;
- extraction status;
- chunks;
- data schema summaries;
- search runs;
- evidence packets;
- answer traces.

Use a separate vector backend for embeddings if needed. The vector backend should store only vector IDs and embeddings; SQLite remains authoritative for metadata and chunk text.

Recommended MVP:

- SQLite for metadata and FTS5 keyword search (unicode61 default tokenizer).
- ChromaDB for vector embeddings, with separate collections per logical index.

## File Classification

Every file gets one classification.

Suggested categories:

```text
document
slides
notebook
code
data_schema
transcript
image_metadata_only
media_metadata_only
archive_metadata_only
binary_metadata_only
installer_metadata_only
model_metadata_only
unknown_metadata_only
```

Suggested extension mapping:

```text
document: .pdf, .docx, .doc, .txt, .md
slides: .pptx, .ppt
notebook: .ipynb
code: .py, .r, .cpp, .h, .m
data_schema: .csv, .xlsx, .json, .jsonl, .sqlite, .db
transcript: .vtt
image_metadata_only: .png, .jpg, .jpeg, .tif, .jfif
media_metadata_only: .mp4, .mov, .mkv, .avi, .m4a, .wav
archive_metadata_only: .zip, .rar, .7z
installer_metadata_only: .exe, .msi, .cab
model_metadata_only: .bin, .joblib, .weights, .tflite, .pt, .pkl, .rdata, .rds
```

The classifier should record `reason_not_indexed` for metadata-only categories.

## SQLite Schema

The schema below is the MVP implementation contract. Feature specs may clarify naming, constraints, or indexes, but should not redesign these tables without updating this architecture document and the decision log.

### courses

One row per direct child folder under `Courses`.

```sql
CREATE TABLE courses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL UNIQUE,
    file_count INTEGER NOT NULL DEFAULT 0,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### files

One row per file under `Courses`, including files that are never indexed.

```sql
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    course_id INTEGER REFERENCES courses(id),
    path TEXT NOT NULL UNIQUE,
    relative_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at TEXT,
    content_hash TEXT,
    category TEXT NOT NULL,
    index_status TEXT NOT NULL,
    reason_not_indexed TEXT,
    discovered_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX idx_files_course_id ON files(course_id);
CREATE INDEX idx_files_extension ON files(extension);
CREATE INDEX idx_files_category ON files(category);
CREATE INDEX idx_files_index_status ON files(index_status);
CREATE INDEX idx_files_hash ON files(content_hash);
```

`index_status` values:

```text
pending
indexed
metadata_only
failed
skipped
```

### extraction_runs

Tracks ingestion/extraction runs.

```sql
CREATE TABLE extraction_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    files_seen INTEGER NOT NULL DEFAULT 0,
    files_indexed INTEGER NOT NULL DEFAULT 0,
    files_metadata_only INTEGER NOT NULL DEFAULT 0,
    files_failed INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
```

### extracted_documents

One row per file with successful extraction or metadata-only summary.

```sql
CREATE TABLE extracted_documents (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    extraction_run_id INTEGER REFERENCES extraction_runs(id),
    extractor_name TEXT NOT NULL,
    extractor_version TEXT,
    status TEXT NOT NULL,
    text_length INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    error TEXT,
    extracted_at TEXT NOT NULL,
    UNIQUE(file_id, extractor_name)
);

CREATE INDEX idx_extracted_documents_file_id ON extracted_documents(file_id);
```

### chunks

Atomic retrieval units. A chunk may come from a PDF page, slide, notebook cell, code section, transcript timestamp, or data schema summary.

`chunks.source_type` is a logical retrieval category, not the original file extension. Store only these values:

```text
document
slides
notebook
code
data_schema
transcript
```

Use `files.extension` for the original extension such as `.pdf`, `.pptx`, `.ipynb`, `.py`, `.csv`, or `.vtt`.

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    extracted_document_id INTEGER REFERENCES extracted_documents(id),
    chunk_uid TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    token_count INTEGER,
    location_type TEXT,
    location_value TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_chunks_file_id ON chunks(file_id);
CREATE INDEX idx_chunks_source_type ON chunks(source_type);
CREATE INDEX idx_chunks_location ON chunks(location_type, location_value);
```

Examples:

```text
source_type=document, location_type=page, location_value=12
source_type=slides, location_type=slide, location_value=8
source_type=notebook, location_type=notebook_cell, location_value=23
source_type=code, location_type=function, location_value=train_model
source_type=data_schema, location_type=schema, location_value=Sheet1
source_type=transcript, location_type=timestamp, location_value=00:12:34
```

### chunk_fts

SQLite FTS5 table for keyword search.

`chunk_fts` is a denormalized search projection, not an external-content FTS table over `chunks`. SQLite remains authoritative through `chunks`, `files`, and `courses`; `chunk_fts` stores the searchable text needed for keyword ranking plus the `chunk_id` needed to join back to authoritative rows.

```sql
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    title,
    course_name,
    file_path,
    source_type UNINDEXED,
    tokenize='unicode61'
);
```

The app should populate this table from a joined projection of `chunks`, `files`, and `courses`:

```sql
SELECT
    chunks.id AS chunk_id,
    chunks.text,
    chunks.title,
    courses.name AS course_name,
    files.path AS file_path,
    chunks.source_type
FROM chunks
JOIN files ON files.id = chunks.file_id
LEFT JOIN courses ON courses.id = files.course_id;
```

For the MVP, keyword indexing may rebuild `chunk_fts` from this projection. Incremental synchronization can be added later, but it must preserve this denormalized search contract.

### embeddings

Maps chunks to embeddings in the selected vector backend.

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id),
    vector_backend TEXT NOT NULL,
    vector_collection TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedded_at TEXT NOT NULL,
    UNIQUE(vector_backend, vector_collection, vector_id),
    UNIQUE(chunk_id, embedding_model)
);

CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
```

### data_summaries

Stores summaries for tabular/semi-structured data without embedding full datasets.

```sql
CREATE TABLE data_summaries (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    format TEXT NOT NULL,
    row_count INTEGER,
    column_count INTEGER,
    table_count INTEGER,
    sheet_count INTEGER,
    schema_json TEXT NOT NULL,
    sample_json TEXT,
    summary_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_id)
);
```

### search_runs

One row per user query retrieval attempt.

```sql
CREATE TABLE search_runs (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT,
    router_output_json TEXT NOT NULL,
    searched_courses_json TEXT NOT NULL,
    searched_indexes_json TEXT NOT NULL,
    keyword_terms_json TEXT NOT NULL,
    semantic_queries_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    weaknesses_json TEXT,
    error TEXT
);
```

### search_results

Stores ranked retrieval results before final packet assembly.

```sql
CREATE TABLE search_results (
    id INTEGER PRIMARY KEY,
    search_run_id INTEGER NOT NULL REFERENCES search_runs(id),
    chunk_id INTEGER REFERENCES chunks(id),
    file_id INTEGER REFERENCES files(id),
    retrieval_method TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL,
    selected_for_evidence INTEGER NOT NULL DEFAULT 0,
    result_json TEXT
);

CREATE INDEX idx_search_results_run_id ON search_results(search_run_id);
CREATE INDEX idx_search_results_selected ON search_results(selected_for_evidence);
```

`retrieval_method` examples:

```text
metadata
keyword
semantic
file_read
notebook_inspection
data_summary
```

### evidence_packets

Stores the exact evidence contract passed to the answer generator.

```sql
CREATE TABLE evidence_packets (
    id INTEGER PRIMARY KEY,
    search_run_id INTEGER NOT NULL REFERENCES search_runs(id),
    packet_json TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
```

### answers

Stores final answer traces.

```sql
CREATE TABLE answers (
    id INTEGER PRIMARY KEY,
    evidence_packet_id INTEGER NOT NULL REFERENCES evidence_packets(id),
    answer_text TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    limitations_json TEXT,
    model_name TEXT,
    created_at TEXT NOT NULL
);
```

## Logical Indexes

Keep indexes logically separate even if stored in the same physical database.

```text
metadata_index: all files and course folders; no chunk source_type
document -> document_index: PDF/DOCX/DOC/TXT/MD chunks
slides -> slides_index: PPTX/PPT slide chunks
notebook -> notebook_index: IPYNB markdown/code cells
code -> code_index: PY/R/CPP/H/M code sections
data_schema -> data_schema_index: CSV/XLSX/JSON/JSONL/SQLite/DB summaries
transcript -> transcript_index: VTT transcript chunks
```

The router chooses candidate logical indexes for each query.

Each logical chunk `source_type` maps to one logical index, and each logical index maps to a separate ChromaDB collection. The router selects which collections to search per query. Cross-index queries search multiple collections and merge results.

## Ingestion Pipeline

### Stage 1: Inventory

Inputs:

- `Courses` root path.

Outputs:

- `courses` rows.
- `files` rows.
- category and index status for each file.

Rules:

- direct child folder name is the course name;
- preserve exact path spellings;
- do not mutate course files;
- use metadata-only status for images, archives, media, binaries, installers, and model artifacts.

### Stage 2: Extraction

Inputs:

- files where `index_status=pending` and category is extractable.

Outputs:

- `extracted_documents`;
- `chunks`;
- optional `data_summaries`.

Extractor responsibilities:

- preserve source location;
- return structured metadata;
- fail per-file without stopping the whole run;
- record extraction errors.

Extractor-specific rules:

- **PDFs**: PyMuPDF for text extraction. If text yield is very low (likely scanned), fall back to Tesseract OCR via pytesseract. Tesseract is an optional system dependency.
- **PPTX slides**: One chunk per slide. Concatenate all text shapes on the slide. Include slide title as chunk title. Append speaker notes to slide text.
- **Notebooks**: One chunk per cell (markdown or code). Include truncated text outputs (max ~500 chars) appended to code cells. Skip image/binary outputs.
- **Python code**: AST-based extraction using the `ast` module. Extract functions, classes, docstrings, and imports as separate chunks.
- **Other code** (R, C++, MATLAB): Regex-based splitting by function definitions, or whole-file chunks.
- **Data files** (CSV, XLSX, JSON, JSONL, SQLite, DB): Schema + sample rows only (column names, types, row count, first 5 rows). No LLM call.
- **Legacy formats** (.doc, .ppt): Skip for MVP. Mark as failed with reason "legacy format not supported yet."
- **Chunking strategy**: Use natural document boundaries (page, slide, cell, function). Sub-chunk any unit exceeding a max token limit (default ~1000 tokens).

### Stage 3: Keyword Indexing

Inputs:

- `chunks`.

Outputs:

- `chunk_fts`.

Rules:

- keep title, text, course name, and file path searchable;
- rebuild `chunk_fts` from the joined `chunks`/`files`/`courses` projection for the MVP;
- add incremental FTS updates later only if they preserve the same projection fields.

### Stage 4: Embedding

Inputs:

- selected chunk rows.

Outputs:

- vector backend records;
- `embeddings` mapping rows.

Rules:

- embeddings are chunk-level;
- vector IDs map back to SQLite chunk IDs;
- do not embed metadata-only files.

## Query Routing

Router output should be structured:

```json
{
  "query_type": "concept_explanation",
  "candidate_courses": ["Information Retrieval", "NLP", "Data Mining"],
  "candidate_indexes": ["document_index", "slides_index", "notebook_index"],
  "needs_keyword_search": true,
  "needs_semantic_search": true,
  "needs_file_inspection": true,
  "needs_python": false
}
```

Supported query types:

```text
concept_explanation
course_summary
cross_course_comparison
find_file
assignment_or_project_lookup
code_question
data_question
study_quiz
portfolio_resume
unknown_or_unsupported
```

The router should combine:

- course folder names;
- file names;
- metadata search;
- keyword hits;
- semantic search over course summaries/chunks;
- LLM classification if available.

Routing is two-stage:

1. Fast rule-based pre-filter: match course names, file extensions, and exact terms in the query text.
2. If the rule-based pass is ambiguous or returns no candidates, fall back to LLM classification.

Most queries that mention a course name or file type are resolved without an LLM call.

## Retrieval Flow

Recommended search sequence:

1. Run metadata search for obvious course/file matches.
2. Select candidate courses and indexes.
3. Run keyword search for exact terms and abbreviations.
4. Run semantic search for conceptual matches.
5. Merge and deduplicate results.
6. Merge results using Reciprocal Rank Fusion (RRF): score = 1/(k + rank). Skip reranking for MVP.
7. Inspect exact top chunks/files when useful.
8. Assemble evidence packet.

Default retrieval parameters (configurable):

- `keyword_top_k`: 20
- `semantic_top_k`: 20
- `final_top_k`: 10 (after merge and deduplication)

## Evidence Packet Schema

The packet should be JSON-serializable and stored exactly as passed to the answer generator.

```json
{
  "query": "Explain MapReduce from my courses",
  "interpreted_intent": "concept_explanation",
  "searched": {
    "courses": ["High Preformance Computing for Big Data", "Data Eng"],
    "indexes": ["slides_index", "document_index", "notebook_index"],
    "keyword_terms": ["mapreduce", "map reduce", "hadoop"],
    "semantic_queries": [
      "MapReduce programming model",
      "Hadoop distributed computation"
    ]
  },
  "evidence": [
    {
      "course": "High Preformance Computing for Big Data",
      "file_id": 123,
      "chunk_id": 456,
      "file": "D:\\Projects\\Uni RAG Agent\\Courses\\High Preformance Computing for Big Data\\...",
      "source_type": "slides",
      "location": "slide 14",
      "text": "Relevant extracted chunk text.",
      "score": 0.82,
      "retrieval_method": "hybrid"
    }
  ],
  "weaknesses": [
    "Videos were not searched because transcripts are not indexed.",
    "Images are metadata-only by design."
  ],
  "answer_constraints": [
    "Answer only from evidence.",
    "Cite course and file.",
    "If evidence is insufficient, say so."
  ]
}
```

## Answering Rules

The answer generator must:

- use only the packet's evidence;
- cite file and location for supported claims;
- distinguish direct evidence from inference;
- state when evidence is insufficient;
- report weak retrieval when weaknesses are present;
- never cite files not present in the packet;
- use structured inline citations with a references section;
- truncate lowest-scoring evidence if total tokens exceed the LLM context window.

## Tool Interfaces

Initial internal tools:

```python
def list_courses() -> list[dict]: ...
def search_metadata(query: str, filters: dict | None = None) -> list[dict]: ...
def keyword_search(query: str, course: str | None = None, indexes: list[str] | None = None, top_k: int = 20) -> list[dict]: ...
def semantic_search(query: str, course: str | None = None, indexes: list[str] | None = None, top_k: int = 20) -> list[dict]: ...
def read_file(path: str, max_chars: int | None = None) -> dict: ...
def read_extracted_chunk(chunk_id: int) -> dict: ...
def inspect_notebook(path: str) -> dict: ...
def summarize_csv(path: str) -> dict: ...
def summarize_xlsx(path: str) -> dict: ...
def summarize_sqlite(path: str) -> dict: ...
def explain_search_coverage(search_run_id: int) -> dict: ...
```

Tools should be exposed through LangChain tool interfaces for integration with the agent framework.

`python_repl()` can exist later, but should be constrained to safe inspection tasks and should not run old course code automatically.

## Safety Rules

- Do not execute course files by default.
- Do not load pickle/joblib/model artifacts by default.
- Do not run installers or archives.
- Do not mutate files under `Courses`.
- Do not transcribe media or OCR images without explicit opt-in.
- Do not answer from memory when the evidence packet lacks support.

## MVP Milestones

1. Inventory and SQLite schema.
2. File classification and skip reasons.
3. Text extraction for documents/slides/notebooks/code/transcripts.
4. Data schema summarization.
5. Keyword search.
6. Vector search.
7. Query router.
8. Evidence packet builder.
9. Answer generator with citations.
10. Evaluation set and retrieval quality checks.
