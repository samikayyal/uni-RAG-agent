# Architecture & Product Decisions

This document records the key decisions made during the design and development of Uni RAG Agent.

## Summary of Decisions

| ID | Decision Title | Status | Date |
| :---: | :--- | :---: | :--- |
| **DEC-001** | Build a course archive intelligence system, not generic folder chat | `Accepted` | 2026-06-21 |
| **DEC-002** | Ignore images for RAG and keep them metadata-only | `Accepted` | 2026-06-21 |
| **DEC-003** | Use selective ingestion instead of embedding everything | `Accepted` | 2026-06-21 |
| **DEC-004** | Separate retrieval/research from final answering with an evidence packet | `Accepted` | 2026-06-21 |
| **DEC-005** | Use hybrid retrieval across metadata, keyword, and semantic indexes | `Accepted` | 2026-06-21 |
| **DEC-006** | Keep binaries, archives, installers, and model artifacts metadata-only | `Accepted` | 2026-06-21 |
| **DEC-007** | Defer video/audio transcription and index only existing transcripts initially | `Accepted` | 2026-06-21 |
| **DEC-008** | Restrict Python/code execution tools by default | `Accepted` | 2026-06-21 |
| **DEC-009** | Use `uv` for Python dependency and run workflows | `Accepted; amended by DEC-031` | 2026-06-21 |
| **DEC-010** | Use LangChain as core framework | `Accepted` | 2026-06-21 |
| **DEC-011** | Use ChromaDB with separate collections per logical index | `Accepted; amended by DEC-031` | 2026-06-21 |
| **DEC-012** | Hybrid chunking for MVP, defer semantic chunking | `Accepted` | 2026-06-21 |
| **DEC-013** | Two-stage query routing with rule-based pre-filter and LLM fallback | `Superseded by DEC-033` | 2026-06-21 |
| **DEC-014** | Skip reranking for MVP, use Reciprocal Rank Fusion for score merging | `Accepted` | 2026-06-21 |
| **DEC-015** | PyMuPDF with Tesseract OCR fallback for scanned PDFs | `Accepted` | 2026-06-21 |
| **DEC-016** | AST-based code extraction for Python, regex fallback for other languages | `Accepted` | 2026-06-21 |
| **DEC-017** | FastAPI backend with HTML/JS frontend | `Accepted` | 2026-06-21 |
| **DEC-018** | LangChain built-in memory for multi-turn conversation | `Accepted` | 2026-06-21 |
| **DEC-019** | One chunk per notebook cell with truncated text outputs | `Accepted` | 2026-06-21 |
| **DEC-020** | Structured inline citations with references section | `Accepted` | 2026-06-21 |
| **DEC-021** | Environment variables and .env file for configuration | `Accepted` | 2026-06-21 |
| **DEC-022** | Per-file failure with detailed error logging | `Accepted` | 2026-06-21 |
| **DEC-023** | Hash plus timestamp hybrid for change detection, soft delete for removed files | `Accepted` | 2026-06-21 |
| **DEC-024** | Skip legacy .doc and .ppt formats for MVP | `Accepted` | 2026-06-21 |
| **DEC-025** | Schema plus sample rows for data file summarization | `Accepted` | 2026-06-21 |
| **DEC-026** | Early hand-curated evaluation set of 15-20 questions | `Accepted` | 2026-06-21 |
| **DEC-027** | Use read-only EDA notebooks for generated app data | `Accepted` | 2026-06-26 |
| **DEC-028** | Null stale search-result chunk references on chunk deletion | `Accepted` | 2026-06-27 |
| **DEC-029** | Exclude non-current source files from retrieval indexes | `Accepted` | 2026-06-30 |
| **DEC-030** | Fake-default embeddings with optional real models and model-namespaced vector collections | `Superseded by DEC-031` | 2026-07-01 |
| **DEC-031** | Real-only production models with injected test doubles | `Accepted` | 2026-07-11 |
| **DEC-034** | Persisted evidence builds with authoritative immutable packets | `Accepted` | 2026-07-13 |

---

## DEC-001: Build a course archive intelligence system, not generic folder chat

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The archive is large and mixed: documents, slides, notebooks, source code, videos, datasets, image-heavy project folders, model artifacts, installers, archives, and binaries. A generic vector database over the whole folder would be noisy, slow, expensive, and hard to trust.

### Decision

Design the product as a course archive intelligence system. It should classify questions, choose relevant courses and indexes, retrieve evidence, inspect files when needed, and answer with citations and search coverage.

### Consequences

The system needs explicit ingestion rules, metadata, search coverage tracking, and evidence packets. This increases design discipline, but makes the answers debuggable and safer.

---

## DEC-002: Ignore images for RAG and keep them metadata-only

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The file scan found `18,633 .png`, `7,840 .jpg`, `92 .jpeg`, and `47 .tif` files. The user confirmed that about 99% of images are data and not useful course knowledge.

### Decision

Do not OCR, caption, embed, or semantically index standalone image files by default. Store image files in the metadata inventory only. Scanned-PDF OCR is a separate PDF extraction fallback covered by DEC-015.

### Consequences

The system can still answer metadata questions about image-heavy folders and datasets. It will not answer from standalone image contents unless a later opt-in OCR/captioning workflow is added for selected folders.

---

## DEC-003: Use selective ingestion instead of embedding everything

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The archive is approximately `27,978` files and `24.4 GB`. It includes useful course documents but also huge artifacts such as a `3.39 GB` word vector `.bin`, a `2.39 GB` `.joblib`, a `1.82 GB` Oracle `.cab`, archives, videos, and model weights.

### Decision

Index only file extensions that are likely to contain useful course knowledge:

```text
.pdf, .pptx, .ppt, .docx, .doc, .txt, .md, .ipynb,
.py, .r, .cpp, .h, .m, .csv, .xlsx, .json, .jsonl,
.sqlite, .db, .vtt
```

Other files are stored as metadata only unless explicitly handled by a later tool.

### Consequences

Ingestion is faster and retrieval is cleaner. The system must record `reason_not_indexed` so skipped files are still explainable.

---

## DEC-004: Separate retrieval/research from final answering with an evidence packet

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The proposed workflow included one agent deciding courses/indexes and using tools, then another agent answering from the gathered information. The useful part is not having "fresh context"; the useful part is an auditable contract between research and answering.

### Decision

Use a structured evidence packet as the boundary between retrieval/research and answer generation. The answer generator may only use evidence in the packet.

### Consequences

The project needs a stable packet schema with searched courses, searched indexes, keyword terms, semantic queries, source chunks, scores, citations, and weaknesses. This reduces hallucination and makes weak retrieval visible.

---

## DEC-005: Use hybrid retrieval across metadata, keyword, and semantic indexes

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

University materials contain exact terms, abbreviations, file-name hints, course names, and semantic concepts. Pure vector search can miss exact terms such as `3NF`, `BM25`, `NPCompleteness`, or `HDFS`; pure keyword search can miss paraphrases.

### Decision

Use metadata filtering, keyword/BM25 search, semantic vector search, and RRF result merging for the MVP. Query planning may use an LLM, but retrieval execution remains deterministic: the application applies planned hard filters and invokes all three search methods before RRF. Reranking is a later optimization only if evaluation shows RRF is insufficient.

### Consequences

The retrieval layer is more complex but much more reliable. Search runs must log enough detail to explain what was searched.

---

## DEC-006: Keep binaries, archives, installers, and model artifacts metadata-only

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The archive contains `.bin`, `.joblib`, `.cab`, `.weights`, `.tflite`, `.pt`, `.pkl`, `.exe`, `.msi`, `.zip`, `.rar`, and `.7z` files. Some formats can be unsafe to load and many are not useful for Q&A.

### Decision

Treat these files as metadata-only by default. Do not load, execute, decompress, or embed them during standard ingestion.

### Consequences

The system avoids unsafe deserialization and irrelevant indexing. Later specialized tools can inspect selected artifacts only after explicit user approval.

---

## DEC-007: Defer video/audio transcription and index only existing transcripts initially

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The scan found lecture and project media such as `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4a`, and `.wav`, plus a few `.vtt` files. Full transcription could be slow and expensive.

### Decision

Index existing `.vtt` transcript files. Store audio/video media as metadata only. Add opt-in transcription later if needed.

### Consequences

The MVP avoids heavy processing. Answers should explicitly say videos were not searched when no transcript was indexed.

---

## DEC-008: Restrict Python/code execution tools by default

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The agent may need tools like `python_repl()`, `read_file()`, `inspect_notebook()`, and data summarizers. However, old course code may install dependencies, mutate files, assume unavailable environments, or execute unsafe logic.

### Decision

Use Python tools only for safe inspection tasks by default: parsing notebooks, summarizing schemas, counting files, and analyzing extracted text. Do not automatically run course scripts, notebooks, installers, or deserialize pickle/joblib artifacts.

### Consequences

The system remains safer and more predictable. Any execution of old project code should be explicit, scoped, and user-approved.

---

## DEC-009: Use `uv` for Python dependency and run workflows

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The project is Python-based, and the user specified the project rule: use `uv` for everything in Python projects.

### Decision

Use `uv add package_name` for dependencies and `uv run ...` for execution.

### Consequences

Documentation, scripts, and commands should assume `uv`. Avoid ad-hoc package installation or direct interpreter execution unless there is a clear reason.

**Amendment:** DEC-031 governs the current model/dependency setup: the base
installation supports non-vector Features 01-06, while vector indexing uses the
optional Hugging Face extra and a reviewed configured profile.

---

## DEC-010: Use LangChain as core framework

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The system needs multi-provider LLM and embedding support (OpenAI, Gemini, Anthropic, Ollama). Options considered: LiteLLM, LangChain/LlamaIndex, thin custom abstraction, or OpenRouter.

### Decision

Use LangChain as the core framework for LLM abstraction, embedding abstraction, ChromaDB integration, and conversation memory.

### Consequences

LangChain becomes a foundational dependency. All LLM calls, embedding pipelines, and retrieval chains use LangChain interfaces. Provider switching is a configuration change. Production model/provider values remain optional until their consumers are implemented; automated tests inject deterministic doubles at model-loader or chat-model boundaries.

---

## DEC-011: Use ChromaDB with separate collections per logical index

* **Status**: Accepted; affected model/collection guidance amended by DEC-031
* **Date**: 2026-06-21

### Context

Vector store candidates were Chroma, LanceDB, Qdrant, FAISS, and sqlite-vss. The architecture defines 7 logical indexes (document, slides, notebook, code, data_schema, transcript, metadata). Physical storage mapping needed to be decided.

### Decision

Use ChromaDB as the vector store. Create separate ChromaDB collections per logical index (e.g. `document_index`, `slides_index`, `code_index`). The query router selects which collections to search per query.

### Consequences

ChromaDB is LangChain-native and simple to set up with disk persistence. Separate collections give clean isolation but cross-index queries require multiple searches. Collection dimensionality depends on the configured embedding model, and tests should use a deterministic injected embedding double.

**Amendment:** DEC-031 replaces the former provider-selection guidance in this
decision's historical text with reviewed real production profiles, explicit model
selection, lazy optional dependencies, runtime dimension probing, and test-only
injection at the model-loader boundary.

---

## DEC-012: Hybrid chunking for MVP, defer semantic chunking

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

Chunking strategy options: natural document boundaries, fixed-size token chunks, hybrid (natural boundaries + sub-chunking), or semantic chunking (LLM/embedding-based topic detection). Semantic chunking would be highest quality but is slow and expensive for thousands of files.

### Decision

Use natural document boundaries as the primary chunking strategy for MVP: one chunk per PDF page, one per slide, one per notebook cell, one per code function/class (Python) or whole file (other languages), one per VTT timestamp block. Sub-chunk any unit exceeding a configurable max token limit (e.g. 1000 tokens). Upgrade to semantic chunking later as an optimization.

### Consequences

Chunking is fast and deterministic with no LLM calls during ingestion. Chunk sizes will vary (a 3-word slide title vs a full PDF page), which may affect retrieval quality. The max-size split catches outliers.

---

## DEC-013: Two-stage query routing with rule-based pre-filter and LLM fallback

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The query router must output structured JSON with query_type, candidate_courses, and candidate_indexes. Options: pure LLM classification, pure rule-based, two-stage hybrid, or embedding-based course matching.

### Decision

Use a two-stage router. Stage 1: fast rule-based pre-filter that matches course names, file extensions, and exact terms in the query text to select candidate courses and indexes. Stage 2: if the rule-based pass is ambiguous or returns no candidates, fall back to LLM classification.

### Consequences

Most queries that mention a course name or file type are resolved without an LLM call, reducing latency and cost. Ambiguous queries like "compare how two courses covered embeddings" still get LLM routing. The rule-based stage needs a list of course names and common aliases.

---

## DEC-014: Skip reranking for MVP, use Reciprocal Rank Fusion for score merging

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

Hybrid retrieval produces results from keyword (BM25) and semantic (cosine similarity) searches on different score scales. Results need to be merged. Reranking with a cross-encoder or LLM could improve quality but adds latency and complexity.

### Decision

Skip reranking for the MVP. Merge keyword and semantic results using Reciprocal Rank Fusion (RRF): each result gets a score of `1/(k + rank)` where `k` is a constant (typically 60). This is rank-based and does not require score normalization. Add a reranker later if retrieval quality is insufficient.

### Consequences

RRF is simple, effective, and widely used in hybrid search systems. It avoids the complexity of normalizing BM25 and cosine scores. No additional model dependency for MVP.

---

## DEC-015: PyMuPDF with Tesseract OCR fallback for scanned PDFs

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

University PDFs vary: text-based, scanned images, exported slides, password-protected. PyMuPDF handles text-based PDFs well but yields no text from scanned documents.

### Decision

Use PyMuPDF as the primary PDF extractor. If text extraction yields little or no text (likely a scanned document), fall back to Tesseract OCR via pytesseract only when `UNI_RAG_OCR_ENABLED` is true and Tesseract is installed. Tesseract is an optional system dependency; extraction works without it but skips scanned PDFs.

### Consequences

Tesseract must be installed separately on Windows (not a pip package). OCR should be optional: if OCR is disabled or Tesseract is not installed, log a warning and mark scanned PDFs as `failed` with reason "scanned PDF, OCR not available." This decision does not permit OCR/captioning of standalone image files; DEC-002 keeps those metadata-only by default.

---

## DEC-016: AST-based code extraction for Python, regex fallback for other languages

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

Code files (`.py`, `.r`, `.cpp`, `.h`, `.m`) need to be chunked. A 500-line Python file with 10 functions is very different from a 20-line R script. Options: AST parsing, tree-sitter, plain text chunking, or skip code files.

### Decision

Use Python's built-in `ast` module for `.py` files to extract functions, classes, docstrings, and imports as separate chunks with `location_type=function` or `location_type=class`. For R, C++, and MATLAB files, fall back to simple regex-based splitting (e.g. function definitions) or whole-file chunks.

### Consequences

Python code gets rich structural metadata. Other languages get basic chunking. No external dependency (tree-sitter) needed. The approach can be upgraded per-language later.

---

## DEC-017: FastAPI backend with HTML/JS frontend

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The evidence packet and citation format is rich (searched courses, weaknesses, file locations). A pure CLI would make this hard to read. Options: CLI only, Streamlit, FastAPI + HTML/JS, Chainlit, or Gradio.

### Decision

Build a FastAPI backend with a simple HTML/JS frontend. This creates a proper API layer that can serve both web and CLI clients.

### Consequences

More work than Streamlit but gives a proper API layer with clear separation of concerns. The frontend can display citations, search coverage, and evidence packets with rich formatting. CLI can also call the same API endpoints.

---

## DEC-018: LangChain built-in memory for multi-turn conversation

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The original plan did not specify conversation memory. Multi-turn context is needed for follow-up questions like "What about in the Data Mining course?" after asking about a concept.

### Decision

Use LangChain's built-in memory modules (e.g. ConversationBufferMemory or similar) to maintain multi-turn context within a session. This integrates naturally with the LangChain framework choice.

### Consequences

The query router must consider conversation history when interpreting follow-up queries. Evidence packets should still be self-contained per query. Memory scope is per-session; no cross-session persistence is needed for MVP.

---

## DEC-019: One chunk per notebook cell with truncated text outputs

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

Jupyter notebooks contain interleaved markdown cells, code cells, and output cells. Chunking options: per-cell, merged cells, whole-notebook conversion, or code/markdown split into different indexes.

### Decision

Create one chunk per notebook cell. Each markdown cell and code cell becomes its own chunk with `location_type=notebook_cell` and the cell index as `location_value`. For code cells, include text outputs appended to the code, truncated to approximately 500 characters. Skip image/binary outputs.

### Consequences

Preserves the notebook's natural structure. Truncated outputs add context (e.g. accuracy scores) without bloating chunks with long tracebacks or dataframe prints.

---

## DEC-020: Structured inline citations with references section

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The answer format affects user experience. Options: inline citations, end-of-answer citations, bullet-point per source, or LLM-decided format.

### Decision

Use structured inline citations: e.g. "MapReduce splits computation into map and reduce phases [High Performance Computing, Lecture5.pptx, slide 14]." Include a references section at the bottom listing all cited files with full paths and locations.

### Consequences

Inline citations make it easy to trace which claim came from which source. The references section provides a clean summary. The answer generator prompt must enforce this format.

---

## DEC-021: Environment variables and .env file for configuration

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The system needs configuration for API keys, model names, data paths, retrieval parameters, and feature flags.

### Decision

Use environment variables loaded from a `.env` file via `python-dotenv`. The `.env` file is gitignored. Store API keys, model selections, paths, and tuning parameters.

### Consequences

Standard practice. Configuration is portable across environments. Secrets stay out of version control.

---

## DEC-022: Per-file failure with detailed error logging

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The archive has ~28,000 files. Some will have corrupted content, unsupported encodings, password protection, or other extraction issues.

### Decision

Fail per-file: if extraction fails on one file, log the error with the file path, error type, and traceback, then continue processing. Show a summary at the end with success, failure, and skip counts plus a list of failed files. Use Python stdlib logging with structured JSON lines output to file and console.

### Consequences

A corrupted PDF does not block 5000 other files. Detailed error logs make diagnosis straightforward. Sequential processing with tqdm-style progress logging keeps the MVP simple.

---

## DEC-023: Hash plus timestamp hybrid for change detection, soft delete for removed files

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

When re-running inventory/extraction, the system needs to detect new, changed, and deleted files without re-processing the entire archive.

### Decision

Use a hash + timestamp hybrid for change detection: check the file's modified timestamp first (fast); if the timestamp has changed, compute the content hash (SHA-256) to confirm whether the content actually changed. For deleted files, use soft delete: mark files as missing (update `last_seen_at`) but keep their rows, chunks, and embeddings. Provide a manual "purge" command to hard-delete old entries.

### Consequences

Avoids re-hashing unchanged files. Soft delete preserves history and avoids orphaned vector IDs in ChromaDB. The purge command gives explicit control over cleanup.

---

## DEC-024: Skip legacy .doc and .ppt formats for MVP

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The file classification lists `.doc` and `.ppt` as extractable, but `python-docx` cannot read `.doc` files and `python-pptx` cannot read `.ppt` files. Handling them requires LibreOffice conversion, `textract`, or COM automation.

### Decision

Skip legacy `.doc` and `.ppt` files for MVP. Classify them as extractable but mark as `failed` with reason "legacy format not supported yet." Prioritize modern formats (`.docx`, `.pptx`) which likely cover most content.

### Consequences

Some older course materials may not be indexed. The system transparently reports these as unsupported. Legacy format support can be added later via LibreOffice conversion or similar.

---

## DEC-025: Schema plus sample rows for data file summarization

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

CSV, XLSX, JSON, and SQLite files in the archive are mainly assignment outputs and training data. Full embedding would be noisy. Options ranged from schema-only to LLM-generated descriptions.

### Decision

Summarize data files with schema and sample rows only: column names, inferred types, row count, and the first 5 rows as text. No LLM call needed. The summary text stored in the chunk is sufficient for keyword and semantic search to match relevant datasets.

### Consequences

Fast and deterministic. No LLM cost for data summarization. Users can find datasets by column names, data characteristics, or row counts.

---

## DEC-026: Early hand-curated evaluation set of 15-20 questions

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The original roadmap placed evaluation in Phase 6, after all implementation. Early evaluation helps catch retrieval quality issues incrementally.

### Decision

Create a small hand-curated evaluation set of 15-20 questions early in development. Cover each query type: concept explanation, file finding, cross-course comparison, code question, data question, and unknown/unsupported. Each question should list expected source files or courses. Run the eval set manually during development as each phase completes.

### Consequences

Retrieval quality can be tracked incrementally. Issues with chunking, indexing, or routing are caught early rather than after the full system is built.

---

## DEC-027: Use read-only EDA notebooks for generated app data

* **Status**: Accepted
* **Date**: 2026-06-26

### Context

After Feature 03, `inventory run` creates useful structured metadata in `data/uni_rag.sqlite`: course rows, file rows, category/status counts, skip reasons, run history, and the extraction backlog. Later stages will create similarly inspectable artifacts: extraction rows, chunks, data summaries, keyword indexes, embeddings, search runs, evidence packets, answers, and evaluation reports. CLI summaries are useful for quick checks, but exploratory analysis needs richer slicing before extraction, indexing, retrieval, and answer quality are tuned.

### Decision

Keep project-owned EDA notebooks under `notebooks/`. Use pandas for DataFrame-oriented notebook analysis and matplotlib-backed pandas plots for lightweight diagnostic charts over generated app data. Add notebooks at stage boundaries where generated artifacts benefit from human inspection:

```text
notebooks/inventory_eda.ipynb
notebooks/extraction_eda.ipynb
notebooks/data_schema_eda.ipynb
notebooks/keyword_index_eda.ipynb
notebooks/vector_index_eda.ipynb
notebooks/retrieval_eda.ipynb
notebooks/answering_eda.ipynb
notebooks/evaluation_eda.ipynb
```

Only create a notebook when its producing stage is implemented and its source artifacts exist. The first implemented notebook is `notebooks/inventory_eda.ipynb`, which reads the SQLite inventory output from `data/uni_rag.sqlite` after `uv run -m uni_rag_agent inventory run`.

EDA notebooks are read-only analysis companions. They may inspect generated app data such as SQLite tables and run artifacts, but they are not the application pipeline and must not mutate `Courses`, write to SQLite, execute course scripts, execute course notebooks, load unsafe artifacts, or become the only place where production behavior exists.

When an implementation change modifies a notebook's source command, source tables, JSON artifact shape, status vocabulary, plots, or interpretation rules, update that notebook in the same change. Notebook outputs and execution counts should be cleared before commit unless a future decision explicitly accepts committed output snapshots.

### Consequences

Notebook analysis can guide extractor priority, performance tuning, and data-quality checks without expanding the runtime surface area. Pandas and matplotlib are accepted for this EDA layer. Additional notebook-specific dependencies should not be added casually unless a later decision explicitly accepts them.

---

## DEC-028: Null stale search-result chunk references on chunk deletion

* **Status**: Accepted
* **Date**: 2026-06-27

### Context

Feature 04 re-extraction deletes stale chunks for files whose content changes, then writes fresh chunks. The planned retrieval schema already has `search_results.chunk_id` references to `chunks.id`. Without an explicit delete policy, a historical search result that points at an old chunk can block re-extraction with a SQLite foreign-key failure.

### Decision

Use `ON DELETE SET NULL` for `search_results.chunk_id`. Historical search result rows are retained, but their obsolete chunk pointer becomes `NULL` when stale chunks are deleted. `search_results.file_id` remains available for source-file traceability, and evidence packets store the exact evidence payload separately.

### Consequences

Changed-file re-extraction can replace chunks without being blocked by historical retrieval rows. Future retrieval and EDA code must treat `search_results.chunk_id` as nullable and should join chunk details with a left join when inspecting historical results. Historical rows whose chunks were deleted can still identify the source file and retrieval metadata, but they cannot rehydrate the exact old chunk text from `chunks`.

---

## DEC-029: Exclude non-current source files from retrieval indexes

* **Status**: Accepted
* **Date**: 2026-06-30

### Context

Inventory and extraction preserve historical SQLite rows so reruns can explain
missing files, skipped files, failures, and stale chunks. That history is useful
for diagnostics, but retrieval should represent the currently indexed corpus.

### Decision

Current retrieval indexes exclude chunks whose joined source file does not have
`files.index_status = 'indexed'`. Feature 06 keyword indexing rebuilds
`chunk_fts` only from current indexed files, and keyword search reapplies the
same filter when joining FTS rows back to `chunks`, `files`, and `courses`.
Future retrieval indexes should follow the same current-file-only policy unless
a later decision creates an explicit historical-search mode.

### Consequences

Missing, skipped, failed, pending, and metadata-only source files do not leak into
normal answers even if historical chunks or stale FTS rows remain in SQLite.
Operational notebooks and storage diagnostics may still inspect those historical
rows, but answer-time retrieval uses only the current indexed corpus.

---

## DEC-030: Fake-default embeddings with optional real models and model-namespaced vector collections

* **Status**: Superseded by DEC-031
* **Date**: 2026-07-01

### Context

Feature 07 adds ChromaDB vector indexing and semantic search through LangChain
embedding abstractions (DEC-010, DEC-011). LangChain's Hugging Face integration
(`langchain-huggingface`) depends on `sentence-transformers`, which pulls in
`transformers` and `torch`. Making that mandatory would make every CI/test
install heavy and would push tests toward network access and model downloads,
which Spec 07 forbids. The system also needs to experiment with more than one
real embedding model over time without mixing incompatible vector spaces, and
re-extraction deletes and replaces stale chunks that already have embedding
mapping rows.

### Decision

- Keep deterministic fake embeddings as the default, offline test path. The fake
  adapter is dependency-light and sized by `UNI_RAG_EMBEDDING_DIM`.
- Ship `chromadb` and `langchain-core` as core dependencies, but put real Hugging
  Face local-model support (`langchain-huggingface` plus the Sentence
  Transformers stack) in an optional `embeddings` extra, imported lazily only
  when a real Hugging Face profile is selected.
- Resolve the embedding model from config by default; an explicit `--model`
  selects a known real profile and overrides `UNI_RAG_USE_FAKE_EMBEDDINGS=true`
  for that command. When fake embeddings are disabled, the configured model must
  resolve to a known real profile or the command fails clearly with a
  `VectorIndexError`/`SemanticSearchError`.
- Support side-by-side models through model-namespaced physical ChromaDB
  collections (`<logical_index>__<model_slug>__<hash>`, hash over
  provider/model/dimension/metric) while the logical collections stay stable.
  Collections use cosine distance. The physical collection is the canonical
  profile identity; fake embeddings always use `fake-embedding` rather than a
  configured real-model name, and SQLite permits one mapping per chunk/profile.
- Use `ON DELETE CASCADE` on `embeddings.chunk_id` so stale embedding mapping
  rows are removed when their chunk is deleted.
- Keep semantic search direct and read-only: it queries ChromaDB, verifies each
  hit against its exact authoritative SQLite mapping, reapplies the
  current-file-only/course/index filters before final top-K truncation, and does
  not persist `search_runs`/`search_results`. Persistence belongs to later
  retrieval/evidence specs.
- Incremental vector sync reconciles the selected Chroma collection with SQLite:
  it removes Chroma-only vectors and stale mappings, restores mappings whose
  vectors disappeared, then embeds missing current chunks.

### Consequences

The default `uv sync` and the whole automated test suite stay offline and
lightweight. Real models are an explicit opt-in (`uv sync --extra embeddings`)
and are exercised only by manual smoke runs. Known real profiles (`BAAI/bge-m3`,
`jinaai/jina-embeddings-v3`, `jinaai/jina-embeddings-v5-text-small`,
`google/embeddinggemma-300m`) live in a registry with provider, dimension, trust,
and gated/access notes; their dependencies and weights are only needed when
selected. Side-by-side profiles never share a physical collection, stale
embedding rows cannot point at deleted chunks, and drift between Chroma and
SQLite is reconciled before incremental indexing.

---

## DEC-031: Real-only production models with injected test doubles

* **Status**: Accepted
* **Date**: 2026-07-11

### Context

The implemented vector feature previously exposed a deterministic runtime
embedding path so the base installation and automated tests could run without
the optional Hugging Face stack. That path made the production contract
ambiguous: a vector command could silently select a non-production model and
configuration reported invented provider/model defaults. The future routing and
answering features also need optional production LLM settings without runtime
test providers.

### Decision

- Production vector commands accept only the four reviewed Hugging Face profiles:
  `BAAI/bge-m3`, `jinaai/jina-embeddings-v3`,
  `jinaai/jina-embeddings-v5-text-small`, and
  `google/embeddinggemma-300m`.
- There is no default embedding model. `index vector` and `search semantic`
  require a nonblank `--model` or `UNI_RAG_EMBEDDING_MODEL`, and report the
  supported profile list when selection is missing or unknown.
- `langchain-huggingface` and `sentence-transformers` remain in the optional
  `embeddings` extra. Model construction and runtime dimension probing stay
  lazy, while the probed dimension remains part of collection identity and
  SQLite telemetry.
- Production LLM provider/model settings are nullable and unset by default until
  routing and answering are implemented.
- Automated tests inject deterministic LangChain embedding and chat doubles at
  model-loader boundaries. Test doubles are not production exports, registry
  profiles, CLI options, or configuration values.
- The existing ChromaDB/SQLite reconciliation, model-namespaced collections,
  authoritative SQLite hydration, course filtering, and read-only semantic
  search contracts remain unchanged.

### Consequences

Plain `uv sync` remains sufficient for non-vector Features 01-06. Vector setup
requires `uv sync --extra embeddings`, a reviewed model selection, and any
model-specific access requirements. Automated tests remain offline and exercise
the real ChromaDB and SQLite pipeline through injected model boundaries. Existing
local vector state created under the superseded contract must be cleared and
rebuilt by its owner; this workspace has no generated vector state to migrate.

---

## DEC-032: Read-only routed hybrid retrieval with explicit model and RRF provenance

* **Status**: Superseded by DEC-033
* **Date**: 2026-07-11

### Context

Feature 08 connects the current inventory, FTS5, and ChromaDB slices without
introducing persistence or answering behavior. The routing boundary must remain
deterministic and auditable while supporting ambiguous queries through optional
LangChain providers.

### Decision

- Every `retrieve` invocation requires a reviewed embedding model from `--model`
  or `UNI_RAG_EMBEDDING_MODEL`, including unsupported routes.
- Supported routes run metadata, keyword, and semantic retrieval. Candidate
  courses and logical indexes are hard filters; zero hits are successful
  weaknesses, while any enabled backend/provider failure is fatal.
- Rule routing resolves obvious queries first. Ambiguous or incomplete scope
  uses the configured exact-provider LLM fallback. Missing LLM configuration,
  invalid output, or low confidence returns `unknown_or_unsupported` with no
  searches; provider invocation failures fail the retrieval command.
- Metadata may return file-level rows with `chunk_id = NULL`. Hybrid fusion
  preserves method, semantic-query, source-rank, native-score, and RRF
  contribution provenance, using one-based unweighted RRF without score
  normalization or reranking.
- Search-run/evidence persistence, source-file inspection, execution, and the
  retrieval notebook remain Feature 09 or later concerns.

### Consequences

The result contract can be serialized directly for later persistence and makes
weak coverage visible without mutating SQLite, ChromaDB, or `Courses`. Retrieval
requires the optional embeddings runtime at execution time, while optional LLM
integrations remain lazy behind `uv sync --extra llm`.

---

## DEC-033: Mandatory LLM query planning for read-only hybrid retrieval

* **Status**: Accepted
* **Date**: 2026-07-11

### Context

The Feature 08 rule router duplicated brittle knowledge about course aliases,
intent cues, extensions, and fuzzy matching. It also made the LLM path an
optional fallback even though query interpretation needs structured planning.

### Decision

- Supersede DEC-013's two-stage rule/LLM router and the rule/fallback language
  in DEC-032. Remove aliases, cues, extension routing, fuzzy course routing,
  and all router-named runtime/public contracts.
- Every `retrieve` invocation first calls exactly the configured LangChain
  provider/model to produce and validate a `QueryPlan`. Provider/model settings
  remain nullable during global config loading, but the pair and the optional
  `llm` extra are mandatory at query-planning/retrieval time.
- A supported plan supplies canonical course and logical-index scopes, keyword
  terms, semantic queries, inspection flags, confidence, and a reason. The
  deterministic application then runs metadata, keyword, and semantic search
  with hard planned filters and merges them only with RRF (DEC-014).
- A valid `unknown_or_unsupported` plan has empty scopes, performs no backend
  search, and returns a successful empty run with the LLM reason as a weakness.
  Invalid/low-confidence plans and provider failures are fatal retrieval errors.
- Keep the reviewed embedding-model precondition, read-only Feature 08 scope,
  fatal backend failures, and RRF provenance from DEC-032 unchanged.

### Consequences

Query intent and search scope are consistently represented by one validated
LLM plan instead of two independently evolving mechanisms. Automated tests
inject chat-model doubles at the planner boundary; no runtime fake provider or
prebuilt-plan bypass exists. Non-retrieval commands remain usable without the
LLM dependency.

---

## DEC-034: Persisted evidence builds with authoritative immutable packets

* **Status**: Accepted
* **Date**: 2026-07-13

### Context

Feature 08 deliberately provides a read-only planner and hybrid retriever, but
Feature 10 needs a stable, auditable handoff containing the exact search
coverage and source text used for answering. Reconstructing that handoff later
from mutable chunks, bounded public top-K results, or snippets would lose RRF
provenance and could promote stale corpus data.

### Decision

- Keep `retrieve` read-only. Add `evidence build` as the only persisted
  retrieval workflow and require it to invoke the configured planner exactly
  once with no public prebuilt-plan bypass.
- Persist one validated plan/settings snapshot per run, every complete bounded
  raw result set plus a completion envelope (including successful empty sets),
  the complete deterministic fused RRF ordering, and exactly
  one immutable packet per successful or valid unsupported run.
- Select evidence only from authoritative current indexed chunks whose file,
  course, path, source type, location, and nonblank text still match the fused
  candidate. File-only metadata results remain coverage/audit rows and never
  become synthetic evidence.
- Enforce `final_top_k` and a positive 12,000-token default whole-chunk budget
  without truncating stored text. Invalid stored token counts use the shared
  whitespace estimator; oversized or budget-excluded candidates are reported.
- Planning failures create no run. Backend failures after planning retain
  committed partial raw results as failed audit rows. Packet assembly or corpus
  drift failures create no packet and select no rows.
- Coverage weaknesses are deterministic, exact-deduplicated, and embedded in
  the canonical packet. Conversation contents, credentials, source files,
  Chroma collections, and extracted caches are never persisted or read during
  packet assembly.
- Legacy `router_output_json` is migrated nondestructively to
  `query_plan_json`; packet-per-run uniqueness is enforced after duplicate
  detection.

### Consequences

Feature 09 adds a focused persistence/evidence service and a read-only retrieval
EDA notebook without adding dependencies. Search history is inspectable even
when a later backend fails, while Feature 10 can consume one exact packet
without re-running retrieval or trusting mutable source state.

