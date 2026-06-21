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
| 07 | [Vector Indexing](07-vector-indexing.md) | LangChain embeddings, Chroma collections, test fakes |
| 08 | [Query Routing and Hybrid Retrieval](08-query-routing-and-hybrid-retrieval.md) | Query types, rule router, optional LLM fallback, RRF |
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

## Shared Conventions

- Use `uv add package_name` for dependencies.
- Use `uv run -m uni_rag_agent ...` for project commands.
- Do not document non-`uv` package installation or direct interpreter commands for normal workflows.
- Do not mutate anything under `D:\Projects\Uni RAG Agent\Courses`.
- Store generated metadata, extracted text caches, Chroma persistence, search runs, and debug artifacts under `D:\Projects\Uni RAG Agent\data`.
- Keep `.env` local and ignored. Commit `.env.example`.
- Treat the SQLite schema in `context/architecture.md` as the MVP storage contract.
- Keep LLM and embedding provider/model choices in configuration. Tests must use deterministic fake adapters.
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

## Backlog Boundaries

These are intentionally outside the MVP specs unless a future decision changes scope:

- Image OCR or captioning.
- Full video/audio transcription.
- Knowledge graph construction.
- Portfolio/resume mode beyond evidence-backed answering support.
- Study/quiz mode beyond query-type routing and future extension points.
- Automatic execution of old course code.
- Loading pickle/joblib/model artifacts.
- Reranking beyond RRF.
