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
| **DEC-009** | Use `uv` for Python dependency and run workflows | `Accepted` | 2026-06-21 |
| **DEC-010** | Use LangChain as core framework | `Accepted` | 2026-06-21 |
| **DEC-011** | Use ChromaDB with separate collections per logical index | `Accepted` | 2026-06-21 |
| **DEC-012** | Hybrid chunking for MVP, defer semantic chunking | `Accepted` | 2026-06-21 |
| **DEC-013** | Two-stage query routing with rule-based pre-filter and LLM fallback | `Accepted` | 2026-06-21 |
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

Do not OCR, caption, embed, or semantically index images by default. Store image files in the metadata inventory only.

### Consequences

The system can still answer metadata questions about image-heavy folders and datasets. It will not answer from image contents unless a later opt-in OCR/captioning workflow is added for selected folders.

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

Use metadata filtering, keyword/BM25 search, semantic vector search, and RRF result merging for the MVP. The router should not rely only on an LLM; it should combine course names, file names, extracted headings, keyword hits, and embeddings. Reranking is a later optimization only if evaluation shows RRF is insufficient.

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

---

## DEC-010: Use LangChain as core framework

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

The system needs multi-provider LLM and embedding support (OpenAI, Gemini, Anthropic, Ollama). Options considered: LiteLLM, LangChain/LlamaIndex, thin custom abstraction, or OpenRouter.

### Decision

Use LangChain as the core framework for LLM abstraction, embedding abstraction, ChromaDB integration, and conversation memory.

### Consequences

LangChain becomes a foundational dependency. All LLM calls, embedding pipelines, and retrieval chains use LangChain interfaces. Provider switching is a configuration change. The MVP must define provider/model environment variables and deterministic fake adapters for tests, but should not hardcode a paid or cloud provider as required.

---

## DEC-011: Use ChromaDB with separate collections per logical index

* **Status**: Accepted
* **Date**: 2026-06-21

### Context

Vector store candidates were Chroma, LanceDB, Qdrant, FAISS, and sqlite-vss. The architecture defines 7 logical indexes (document, slides, notebook, code, data_schema, transcript, metadata). Physical storage mapping needed to be decided.

### Decision

Use ChromaDB as the vector store. Create separate ChromaDB collections per logical index (e.g. `document_index`, `slides_index`, `code_index`). The query router selects which collections to search per query.

### Consequences

ChromaDB is LangChain-native and simple to set up with disk persistence. Separate collections give clean isolation but cross-index queries require multiple searches. Collection dimensionality depends on the configured embedding model, and tests should use a deterministic fake embedding implementation.

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

Use PyMuPDF as the primary PDF extractor. If text extraction yields little or no text (likely a scanned document), fall back to Tesseract OCR via pytesseract. Tesseract is an optional system dependency; extraction works without it but skips scanned PDFs.

### Consequences

Tesseract must be installed separately on Windows (not a pip package). OCR should be optional: if Tesseract is not installed, log a warning and mark scanned PDFs as `failed` with reason "scanned PDF, OCR not available."

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

