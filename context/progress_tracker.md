# Progress Tracker

## Current Status

* **Current Phase**: Phase 5: Answering Interface
* **Current Goal**: Begin Feature 11 FastAPI/HTML UI work. Feature 10 post-review hardening is complete with Markdown-aware prose rejection, canonical `chunk:<id>` compatibility aliases, deterministic whole-prompt budgeting with stable retained citation ids and omission limitations, complete packet-relative answering EDA citation diagnostics, 227 passing tests, and healthy storage.

The project now has the Feature 01 foundation package, CLI dispatcher, typed configuration loader, JSONL logging helper, fixture convention, `.env.example`, README developer commands, Feature 02 storage initialization, SQLite MVP schema creation, storage health checks, focused foundation/storage tests, negative-path config/storage diagnostics coverage, pytest discovery constrained to this project's `tests/` directory, Feature 03 inventory/file classification with CLI commands, SQLite upserts, idempotent reruns, timestamp-first/hash-on-change behavior, metadata-only skip reasons, missing-file soft marking, current course-total resets, explicit SQLite connection closing, accurate inventory run metrics, inventory summaries, inventory CLI JSONL run logs, a smaller inventory module split for models, classification vocabulary, and filesystem helpers, a pandas-based read-only inventory EDA notebook under `notebooks/`, Feature 04 text extraction/chunking with CLI commands, per-file extraction failures, source-location chunks, extraction CLI JSONL run logs, Python module-docstring chunk de-duplication, stale chunk/index cleanup on re-extraction, optional scanned-PDF OCR gating, legacy `.doc`/`.ppt` failure reasons, a focused extraction package split for models/constants, chunking, persistence, and grouped format extractors, a pandas/matplotlib read-only extraction EDA notebook with diagnostic plots, a search-results foreign-key policy that nulls stale chunk references when chunks are deleted during re-extraction, Feature 05 data schema summaries for CSV/XLSX/JSON/JSONL/SQLite/DB files with deterministic schema/sample summaries, `data_summaries` persistence, `data_schema` chunks, CLI JSONL run logs, a compact five-file data-summary package split for public API, orchestration/chunk conversion, format readers, persistence, and summary builders/utilities, a pandas/matplotlib read-only data-schema EDA notebook, and Feature 06 keyword indexing with rebuild-first SQLite FTS5 synchronization, current-file-only stale chunk filtering, plain-text OR keyword search, course/logical-index filters, table and JSON CLI output, keyword-index and keyword-search JSONL telemetry, centralized current-chunk eligibility SQL, shared read-only SQLite connection handling, shared retrieval result models, blank-chunk rebuild diagnostics, and a read-only keyword-index EDA notebook. Feature 07 vector indexing now uses only explicitly selected or configured reviewed Hugging Face profiles, loads the optional model stack lazily, probes runtime dimensions, preserves model-namespaced ChromaDB collections and SQLite reconciliation, validates exact vector mappings during read-only semantic search, applies course/index/current-file filters before final top-K, emits normalized model telemetry, and uses injected deterministic test doubles for offline coverage. The real-only hardening pass also keeps test vectors nonzero and sufficiently expressive for Chroma coverage, falls back to exact course-partition scoring if a filtered HNSW query misses every candidate, isolates no-model CLI tests from repository `.env` settings, creates collision-resistant run-log names, validates reviewed-profile/test-double parity and declared profile metadata, and covers `trust_remote_code` propagation. Feature 08 now uses mandatory LLM query planning with validated plans, metadata/keyword/semantic orchestration, RRF provenance models, and a read-only retrieval CLI. Feature 09 adds persisted search runs, result-set completion envelopes, immutable evidence packets, and coverage reporting. Feature 10 adds strict packet-only answer generation, deterministic positional citations and references, separate answer-provider configuration, packet-relative append-only trace validation, planner-only in-memory sessions, `answer`/`ask`, sanitized telemetry, and answering EDA.

## Project Roadmap

- [x] **Phase 0: Design and Context Setup**
  - [x] Profile `Courses` folder file types, size, and major data characteristics.
  - [x] Record core project overview and product direction.
  - [x] Decide to ignore images for RAG and keep them metadata-only.
  - [x] Define researcher/evidence-packet/answerer workflow.
  - [x] Fill context templates.
  - [x] Add technical architecture document.
  - [x] Complete design review: resolve LLM framework, vector store, chunking, routing, interface, and 20+ implementation decisions.
  - [x] Align context docs around accepted stack and MVP boundaries.
  - [x] Add module-level feature specifications under `context/feature-specs/`.
  - [x] Clarify chunk `source_type` as a logical category and map source types to logical indexes.
  - [x] Clarify OCR scope: standalone images stay metadata-only, scanned PDFs may use configured Tesseract fallback.
  - [x] Align Feature 01 foundation package layout and fixture contracts with architecture.

- [x] **Phase 1: Inventory Foundation**
  - [x] Create project package structure.
  - [x] Add dependencies with `uv`.
  - [x] Implement Feature 02 configuration and storage initialization.
  - [x] Create SQLite metadata database.
  - [x] Implement filesystem crawler for `Courses`.
  - [x] Classify files into indexed vs metadata-only categories.
  - [x] Store skip reasons for images, binaries, archives, media, and unsafe artifacts.
  - [x] Add a pandas-based read-only EDA notebook for analyzing SQLite inventory output after `inventory run`.

- [ ] **Phase 2: Text Extraction**
  - [x] Extract text from PDFs (PyMuPDF + optional Tesseract OCR fallback).
  - [x] Extract text from PPTX slides (one chunk per slide with speaker notes).
  - [x] Extract text from DOCX files.
  - [x] Parse TXT and Markdown files.
  - [x] Parse notebooks into one chunk per cell with truncated text outputs.
  - [x] Parse Python code via AST into functions/classes/imports; regex fallback for R/C++/MATLAB.
  - [x] Parse existing VTT transcripts.
  - [x] Generate schema summaries for CSV/XLSX/JSON/JSONL/SQLite/DB files.
  - [x] Add `notebooks/extraction_eda.ipynb` when text extraction lands.
  - [x] Add `notebooks/data_schema_eda.ipynb` when data summaries land.

- [ ] **Phase 3: Indexing and Search**
  - [x] Build chunk table and content metadata.
  - [x] Add keyword search, preferably SQLite FTS5 for MVP.
  - [x] Add vector embedding pipeline with ChromaDB (separate collections per logical index).
  - [x] Keep separate logical indexes for documents, slides, notebooks, code, data schemas, and transcripts.
  - [x] Add search run logging for searched/found/missing reporting (Feature 09 persistence).
  - [x] Add `notebooks/keyword_index_eda.ipynb` when keyword indexing lands.
  - [x] Add `notebooks/vector_index_eda.ipynb` when vector indexing lands.

- [ ] **Phase 4: Retrieval and Evidence Packets**
  - [x] Replace the two-stage router with mandatory LLM query planning.
  - [x] Implement hybrid retrieval over metadata, keyword, and semantic search.
  - [x] Implement Reciprocal Rank Fusion for result merging.
  - [x] Implement evidence packet schema and canonical persistence (Feature 09).
  - [x] Add coverage/weakness reporting with authoritative token-budget selection.
  - [x] Add `notebooks/retrieval_eda.ipynb` for read-only persisted retrieval/evidence traces.

- [ ] **Phase 5: Answering Interface**
  - [x] Implement answer generator that uses only evidence packets.
  - [x] Add citation rendering.
  - [x] Add refusal/insufficient-evidence behavior.
  - [x] Add `notebooks/answering_eda.ipynb` when answer traces are persisted.
  - [ ] Build FastAPI backend with HTML/JS frontend.

- [ ] **Phase 6: Evaluation and Hardening**
  - [ ] Create a hand-curated eval set of 15-20 questions covering each query type (start early, run incrementally).
  - [ ] Add `notebooks/evaluation_eda.ipynb` when evaluation reports are written.
  - [ ] Maintain existing read-only EDA notebooks whenever their source commands, tables, JSON artifacts, status vocabulary, or interpretation rules change.
  - [ ] Measure retrieval quality and citation quality.
  - [ ] Add regression tests for ingestion and evidence packet generation.
  - [ ] Optimize slow filesystem scans.
  - [ ] Keep feature specs updated as implementation decisions change.

- [ ] **Later / Optional**
  - [ ] Opt-in video/audio transcription.
  - [ ] Opt-in standalone image OCR/captioning for selected non-dataset folders.
  - [ ] Knowledge graph over courses, topics, assignments, projects, datasets, and code.
  - [ ] Portfolio/resume mode.
  - [ ] Study/quiz mode.

## Active Task List

- `[ ]` Pending task
- `[/]` In-progress task
- `[x]` Completed task

### Active Tasks

* [x] Fill `context/project_overview.md` from the root project overview.
* [x] Fill `context/decisions.md` with accepted design decisions.
* [x] Fill `context/progress_tracker.md` with roadmap and current status.
* [x] Add `context/architecture.md` with technical storage, schema, and pipeline details.
* [x] Complete design review session resolving framework, vector store, chunking, routing, interface, and implementation decisions.
* [x] Review and patch planning-doc drift before module spec creation.
* [x] Create `context/feature-specs/` with MVP module implementation contracts.
* [x] Clarify logical `source_type` values across architecture and MVP specs.
* [x] Clarify OCR wording across overview, architecture, decisions, and feature specs.
* [x] Align Feature 01 package layout, fixture contract, and logging helper naming with `context/architecture.md`.
* [x] Implement Feature 01 project foundation: package skeleton, CLI stubs, config loader, JSONL logging helper, fixtures, `.env.example`, README, and tests.
* [x] Implement Feature 02 configuration and storage: typed config contract, safe `.env` reporting, generated data directories, SQLite MVP schema, FTS5 check, storage CLI, pytest discovery scoped to `tests/`, positive-path tests, and negative-path diagnostics tests.
* [x] Implement Feature 03 inventory and file classification: course discovery, streaming file crawl, spec category mapping, SQLite course/file upserts, inventory run records, metadata-only reasons, idempotent unchanged-file reruns, hash-on-change behavior, missing-file soft marking, summary CLI, and focused tests.
* [x] Address Feature 03 review findings: close SQLite connections explicitly, reset stale course totals when course folders disappear, keep inventory run `files_indexed` metrics accurate, and add regression tests.
* [x] Add Feature 03 inventory failure-path regression tests for file stat failures, hash failures, nested directory listing diagnostics, and failed root-listing run records.
* [x] Split the large inventory core support code into dedicated models, classification, and filesystem-helper modules while preserving inventory CLI behavior and public package imports.
* [x] Run Brooks full sweep on Features 01-03: fix transient hash-failure recovery, align the Feature 02 runs-dir environment variable contract, and verify the current safety net.
* [x] Add `notebooks/inventory_eda.ipynb` for pandas-based read-only EDA over `data/uni_rag.sqlite` after `inventory run`, and document the notebook strategy across `context/`.
* [x] Expand the notebook roadmap across applicable stages and record that existing notebooks must be updated when their source artifact contracts change.
* [x] Implement Feature 04 text extraction and chunking.
* [x] Address Feature 04 review finding: set stale `search_results.chunk_id` references to `NULL` when deleted chunks are replaced during re-extraction, and add regression coverage.
* [x] Run Brooks full sweep on Features 01-04: add CLI JSONL run logs, avoid duplicating Python module docstrings in generic module chunks, and simplify failed-extraction persistence.
* [x] Fix `notebooks/extraction_eda.ipynb` so its SQLite setup locates the repository root from either the project root or `notebooks/` before opening `data/uni_rag.sqlite` in read-only mode.
* [x] Add matplotlib-backed plots to `notebooks/extraction_eda.ipynb` for extraction outcomes, category/status coverage, chunk coverage, text/token distributions, failure reason counts, and failure hotspots; document matplotlib as the accepted EDA plotting dependency.
* [x] Split the large extraction core into dedicated models/constants, chunking, persistence, and grouped format extractor modules while preserving the public extraction API and CLI behavior.
* [x] Implement Feature 05 data schema summaries.
* [x] Split the large Feature 05 data-summary module into a compact five-file package for public API, orchestration/chunk conversion, format readers, persistence, and summary builders/utilities while preserving the public extraction API.
* [x] Verify and fix Brooks test-quality findings for Features 01-05: extract shared SQLite search-result helpers, add direct extraction/data-summary boundary regressions, label broad smoke-test assertions, and move CLI integration assertions from exact stdout strings to SQLite/run-log state.
* [x] Implement Feature 06 keyword indexing: rebuild `chunk_fts` from current indexed chunks, expose plain-text keyword search with course/logical-index filters, add `index keyword` and `search keyword` CLI commands, keep semantic search as a Feature 07 stub, add focused tests, and add `notebooks/keyword_index_eda.ipynb`.
* [x] Address Feature 06 review findings: centralize current indexed chunk eligibility SQL, remove hardcoded FTS insert placeholders, move read-only SQLite connection handling into storage, add keyword-search JSONL telemetry, make blank chunk skips visible in rebuild diagnostics, and split the broad projection/filter test.
* [x] Implement Feature 07 vector indexing: shared current-file-only eligibility module, deterministic offline fake embeddings plus a known real Hugging Face profile registry behind the optional `embeddings` extra, ChromaDB cosine collections namespaced per embedding model, incremental `index vector` with `--rebuild`, read-only `search semantic` joining vector ids back to SQLite, `RetrievalResult` nullable `vector_collection`/`vector_id`, `embeddings.chunk_id` `ON DELETE CASCADE` migration, vector CLI JSONL telemetry, `notebooks/vector_index_eda.ipynb`, and record DEC-030.
* [x] Address Feature 07 lifecycle review findings before Spec 08: make physical collections the canonical profile identity, keep fake embeddings distinct from configured real models, migrate SQLite mappings to one row per chunk/profile, reconcile missing/orphaned Chroma vectors during sync, reject semantic hits without an exact current mapping, apply course filters before final top-K truncation, and add rollover/drift/filter/error-boundary regressions.
* [x] Remove runtime synthetic model configuration: make production embedding selection explicit and registry-only, make optional LLM settings nullable and unset by default, inject deterministic embedding/chat doubles only in tests, update the vector CLI telemetry and subprocess shim, refresh current documentation/notebooks, and record DEC-031 while preserving the superseded Feature 07 history above.
* [x] Harden the real-only vector test and telemetry contract: prevent deterministic test-vector normalization failures, guarantee course-filter correctness when filtered HNSW returns no candidates, isolate CLI configuration from repository `.env` state, give JSONL run logs unique names, and add profile-registry/loader-option regression coverage.
* [x] Implement Feature 08 read-only hybrid retrieval with mandatory LLM query planning: add validated planning configuration, plural-course/phrase-aware backend search compatibility, deterministic metadata search, RRF provenance models, retrieval orchestration, safe CLI/debug/JSON output, and retrieval telemetry without Feature 09 persistence.
* [x] Replace Feature 08 rule-routing regression coverage with injected-LLM query-planning tests for valid/unsupported plans, validation and provider boundaries, deterministic backend orchestration, zero-hit weaknesses, and deterministic RRF ties.
* [x] Harden Feature 08 test quality: cover metadata/keyword/semantic backend failure boundaries and complete weak-result reports, centralize temporary config/SQLite/subprocess setup, isolate CLI tests from inherited `UNI_RAG_*` variables, and scope the vector loader patch to embedding-dependent tests.
* [x] Implement Feature 09 search-run/evidence persistence: add the 12,000-token configuration, migrate legacy search-run schema safely, persist raw and complete fused results, assemble canonical authoritative packets, expose coverage/loading services, add `evidence build`/`evidence show`, extend safe telemetry, create the retrieval EDA notebook, and add focused lifecycle/budget/drift tests.
* [x] Close Feature 09 review findings: persist successful empty result-set envelopes, require contribution-based chunk-hit coverage and packet/run status consistency, harden credential redaction, and add behavioral CLI/notebook/partial-run regression coverage; verify 187 tests, schema health, help surfaces, compilation, and diff cleanliness.
* [x] Implement Feature 10 answering and citations: strict packet-only answer prompts, positional structured citations and deterministic references, empty-evidence/safe-refusal behavior, bounded retries, separate answer provider/model configuration, append-only answer traces, planner-only `AnswerSession`, `answer`/`ask` CLI commands, telemetry, and `notebooks/answering_eda.ipynb`.
* [x] Close Feature 10 review findings: enforce packet-relative validation at the append-only answer boundary, reject model-authored citation lookalikes and rendered sections, require qualified answer-model identities, remove traceback-bearing `ask` telemetry, and add persistence/CLI/session failure regressions; Brooks follow-up review reports no remaining findings.
* [x] Close the post-commit Feature 10 review findings: reject Markdown-decorated counterfeit sections and link/numeric marker lookalikes, implement explicit `chunk:<id>` alias canonicalization, enforce a configurable complete answer-prompt budget without renumbering retained evidence, and make answering EDA validate parse status plus every authoritative citation/rendering field.
