# Feature Spec 09: Evidence Packets and Coverage

## Purpose

Create the auditable boundary between retrieval and answering: a structured evidence packet that records query interpretation, searched courses/indexes, evidence, scores, citations, and weaknesses.

## Depends On

- [08-query-routing-and-hybrid-retrieval.md](08-query-routing-and-hybrid-retrieval.md)
- `context/architecture.md` tables: `search_runs`, `search_results`, `evidence_packets`
- DEC-004, DEC-005, DEC-020, DEC-022

## In Scope

- Persist search runs and result rows.
- Select final evidence from retrieval results.
- Build JSON-serializable evidence packets.
- Generate searched/found/missing coverage details.
- Generate weaknesses for skipped source types, empty indexes, unsupported formats, and low confidence.
- Store evidence packets exactly as passed to the answer generator.
- Add or update the retrieval/evidence EDA notebook for persisted search and evidence traces.

## Out of Scope

- Writing final natural-language answers.
- Building the UI display.
- Reranking.
- Executing file/code inspection beyond reading already extracted chunks.

## Public Interfaces

Command:

```powershell
uv run -m uni_rag_agent evidence build "Explain MapReduce from my courses"
uv run -m uni_rag_agent evidence show --search-run-id 1
```

Notebook:

```text
notebooks/retrieval_eda.ipynb
```

This notebook is shared with spec 08. Once evidence packets are implemented, it should also inspect `evidence_packets`, selected evidence, packet weaknesses, searched/found/missing coverage, and result-selection behavior.

Internal interfaces:

```python
record_search_run(query: str, router_output: RouterOutput) -> int
record_search_results(search_run_id: int, results: list[RetrievalResult]) -> None
build_evidence_packet(search_run_id: int, results: list[RetrievalResult]) -> EvidencePacket
load_evidence_packet(evidence_packet_id: int) -> EvidencePacket
explain_search_coverage(search_run_id: int) -> SearchCoverage
```

Evidence packet fields:

```text
query
interpreted_intent
searched.courses
searched.indexes
searched.keyword_terms
searched.semantic_queries
evidence[]
weaknesses[]
answer_constraints[]
```

Evidence item fields:

```text
course
file_id
chunk_id
file
source_type
location
text
score
retrieval_method
```

`source_type` in each evidence item must be the logical chunk category (`document`, `slides`, `notebook`, `code`, `data_schema`, or `transcript`). If the UI or answer needs the original file format, derive it from the evidence file path or joined `files.extension` metadata.

## Storage and Schema Impact

Populate:

- `search_runs`
- `search_results`
- `evidence_packets`

Update:

- `search_results.selected_for_evidence`

Evidence packet JSON must be stored exactly as given to the answer generator. Do not store only a summary.

## Workflow

1. Start a `search_runs` row before retrieval or immediately after router output is available.
2. Persist raw retrieval results with method, rank, score, and result JSON.
3. Deduplicate and select final evidence up to `final_top_k`.
4. Load full chunk text and source metadata from SQLite.
5. Build citation-ready evidence items.
6. Generate weaknesses from router/retrieval/index metadata.
7. Store the packet JSON and evidence count.
8. Return the packet to the answer generator.
9. Keep `notebooks/retrieval_eda.ipynb` aligned with `search_runs`, `search_results`, `evidence_packets`, packet JSON shape, weakness semantics, and selected-evidence rules.

## Failure and Safety Rules

- If retrieval returns no evidence, still create a packet with searched fields and weaknesses.
- Do not include evidence text that is not present in stored chunks or safe summaries.
- Do not cite files absent from the packet.
- If total evidence text is too large, select highest-scoring evidence and record that lower-scoring evidence was omitted.
- Do not read or mutate source files under `Courses` during packet assembly.
- The EDA notebook must read generated app data only and must not mutate SQLite, evidence packet JSON, or `Courses`.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated fixture tests build packets from synthetic fixture retrieval results.
- Verify empty retrieval produces a valid insufficient-evidence packet.
- Verify selected results are marked in `search_results`.
- Verify evidence items include course, file, source type, location, text, score, and retrieval method.
- Verify weaknesses include metadata-only images/media when relevant.
- Verify packet JSON round-trips exactly.
- Verify `notebooks/retrieval_eda.ipynb`, once evidence persistence lands, is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Optional smoke: build an evidence packet from a tiny fixture retrieval run.

## Acceptance Criteria

- Evidence packets are self-contained enough for answering.
- Search coverage can explain courses, indexes, keywords, semantic queries, found evidence, and missing coverage.
- Packets are persisted exactly and can be reloaded by ID.
- Answer generation never needs to inspect retrieval internals directly.
- `notebooks/retrieval_eda.ipynb` can inspect persisted evidence and coverage traces without mutating generated or source data.
