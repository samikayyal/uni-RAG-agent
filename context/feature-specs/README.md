# Feature Specs

This folder contains MVP module implementation contracts for Uni RAG Agent.

Use these specs with the rest of `context/` as the operational source of truth:

- `context/project_overview.md` for product boundaries and constraints.
- `context/architecture.md` for storage, schema, pipeline, and interface contracts.
- `context/decisions.md` for accepted technology and behavior decisions.
- `context/progress_tracker.md` for phase status.

The root `project-overview.md` remains the fuller narrative background, but implementation work should follow this folder and `context/architecture.md` first.

## Spec Order

| Order | Spec | Purpose |
| :---: | :--- | :--- |
| 01 | [Project Foundation](01-project-foundation.md) | Package layout, `uv` workflow, commands, logging, fixtures |
| 02 | [Configuration and Storage](02-configuration-and-storage.md) | `.env`, config object, data layout, SQLite, Chroma paths |
| 03 | [Inventory and File Classification](03-inventory-and-file-classification.md) | Course discovery, file rows, categories, skip reasons |
| 04 | [Text Extraction and Chunking](04-text-extraction-and-chunking.md) | Extractors, natural chunks, source locations |
| 05 | [Data Schema Summaries](05-data-schema-summaries.md) | CSV/XLSX/JSON/SQLite summaries and sample rows |
| 06 | [Keyword Indexing](06-keyword-indexing.md) | SQLite FTS5 synchronization and search |
| 07 | [Vector Indexing](07-vector-indexing.md) | Reviewed Hugging Face embeddings, Chroma collections, injected test doubles |
| 08 | [Mandatory LLM Query Planning and Hybrid Retrieval](08-query-routing-and-hybrid-retrieval.md) | Validated LLM query plans, deterministic retrieval, RRF |
| 09 | [Evidence Packets and Coverage](09-evidence-packets-and-coverage.md) | Evidence packet contract, weakness reporting, persistence |
| 10 | [Answering and Citations](10-answering-and-citations.md) | Evidence-only answers, inline citations, references |
| 11 | [FastAPI HTML UI](11-fastapi-html-ui.md) | Answer API, simple UI, operational boundaries |
| 12 | [Evaluation and Hardening](12-evaluation-and-hardening.md) | Eval set, regression checks, smoke runs, performance |

## Dependency Map

```text
01 -> 02
02 -> 03
03 -> 04
03 -> 05
04 -> 06
05 -> 06
04 -> 07
05 -> 07
06 -> 08
07 -> 08
08 -> 09
09 -> 10
10 -> 11
03 -> 12
04 -> 12
08 -> 12
09 -> 12
10 -> 12
```

Implementation can start with specs 01-03, then proceed in dependency order. Specs 04 and 05 can be implemented in parallel after inventory exists. Specs 06 and 07 can also be implemented in parallel once chunks and data summaries exist.

## Notebook Map

EDA notebooks are planned only for stages that produce generated artifacts worth inspecting manually. Create each notebook when its producing feature is implemented; do not add empty placeholder notebooks.

| Specs | Notebook | Status | What It Inspects |
| :--- | :--- | :--- | :--- |
| 02 | None required for MVP | Not applicable | `storage check` is the canonical storage-health workflow until schema migration complexity exists. |
| 03 | `notebooks/inventory_eda.ipynb` | Implemented | `courses`, `files`, inventory run rows, categories, statuses, skip reasons, backlog, freshness. |
| 04 | `notebooks/extraction_eda.ipynb` | Implemented | `extraction_runs`, `extracted_documents`, `chunks`, extraction failures, failure-reason plots, text/chunk coverage. |
| 05 | `notebooks/data_schema_eda.ipynb` | Implemented | `data_summaries`, data-schema chunks, row/column/table/sheet counts, sample coverage. |
| 06 | `notebooks/keyword_index_eda.ipynb` | Planned | `chunk_fts`, keyword coverage, source-type distribution, query smoke checks. |
| 07 | `notebooks/vector_index_eda.ipynb` | Planned | `embeddings`, Chroma collection metadata, embedding model/dimension coverage. |
| 08-09 | `notebooks/retrieval_eda.ipynb` | Implemented | Query plans, `search_runs`, result-set completion envelopes, `search_results`, RRF mix, evidence packets, coverage, token budgets, failures, and weaknesses. |
| 10 | `notebooks/answering_eda.ipynb` | Planned | `answers`, citation validation, limitations, model traces, injected-test behavior. |
| 11 | None required for MVP | Not applicable | UI behavior is covered by API/UI tests; inspect underlying traces through retrieval/answering notebooks. |
| 12 | `notebooks/evaluation_eda.ipynb` | Planned | `data/runs/eval/` reports, retrieval/citation scores, failures, runtime summaries. |

## Shared Conventions

- Use `uv add package_name` for dependencies.
- Use `uv run -m uni_rag_agent ...` for project commands.
- Do not document non-`uv` package installation or direct interpreter commands for normal workflows.
- Do not mutate anything under `D:\Projects\Uni RAG Agent\Courses`.
- Store generated metadata, extracted text caches, Chroma persistence, search runs, and debug artifacts under `D:\Projects\Uni RAG Agent\data`.
- Keep project EDA notebooks under `notebooks/`, use pandas for DataFrame-oriented analysis and matplotlib-backed plots for lightweight diagnostics, make notebooks read generated app data only, and do not use them to mutate SQLite, `Courses`, or source course files.
- Update the relevant notebook when a stage changes the command, tables, JSON artifacts, status vocabulary, plots, or interpretation rules that notebook reads.
- Clear notebook outputs and execution counts before committing unless a future decision explicitly permits committed output snapshots.
- Keep `.env` local and ignored. Commit `.env.example`.
- Treat the SQLite schema in `context/architecture.md` as the MVP storage contract.
- Keep optional LLM settings and the reviewed embedding model choice in configuration. Production providers are real/configured; tests inject deterministic doubles at model-loader or chat-model boundaries.
- Automated tests must use small committed fixtures. Full `Courses` archive checks are optional smoke tests only.

## Standard Spec Shape

Each spec uses these sections:

- Purpose
- Depends On
- In Scope
- Out of Scope
- Public Interfaces
- Storage and Schema Impact
- Workflow
- Failure and Safety Rules
- Tests
- Acceptance Criteria

When a spec has an applicable notebook in the Notebook Map, include the notebook path in Public Interfaces, safety rules, tests, and acceptance criteria for that spec.

## Backlog Boundaries

These are intentionally outside the MVP specs unless a future decision changes scope:

- Standalone image OCR or captioning.
- Full video/audio transcription.
- Knowledge graph construction.
- Portfolio/resume mode beyond evidence-backed answering support.
- Study/quiz mode beyond query-type routing and future extension points.
- Automatic execution of old course code.
- Loading pickle/joblib/model artifacts.
- Reranking beyond RRF.
