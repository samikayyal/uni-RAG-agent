# Feature Spec 08: Query Routing and Hybrid Retrieval

## Purpose

Turn a user query into a structured retrieval plan, run metadata, keyword, and semantic searches over the right courses and logical indexes, then merge results with Reciprocal Rank Fusion (RRF).

## Depends On

- [03-inventory-and-file-classification.md](03-inventory-and-file-classification.md)
- [06-keyword-indexing.md](06-keyword-indexing.md)
- [07-vector-indexing.md](07-vector-indexing.md)
- `context/architecture.md` retrieval flow and router output
- DEC-005, DEC-010, DEC-013, DEC-014

## In Scope

- Support the MVP query types defined in `context/architecture.md`.
- Implement a rule-based router using course names, file extensions, query terms, and index hints.
- Add optional LLM fallback through LangChain only when configured.
- Run metadata, keyword, and semantic searches.
- Merge result lists with RRF.
- Deduplicate by chunk ID and file ID where appropriate.
- Return a retrieval result set ready for evidence packet assembly.
- Add or update the retrieval EDA notebook once retrieval/search traces are persisted by spec 09.

## Out of Scope

- Final answer generation.
- Evidence packet persistence.
- Reranking with cross-encoders or LLMs.
- Automatic execution of Python/course code.
- Full conversation UI.

## Public Interfaces

Command:

```powershell
uv run -m uni_rag_agent retrieve "Explain MapReduce from my courses"
uv run -m uni_rag_agent retrieve "Find my database normalization assignment" --debug
```

Notebook:

```text
notebooks/retrieval_eda.ipynb
```

Create this notebook when specs 08-09 are implemented enough to persist retrieval traces. It should inspect router outputs, `search_runs`, `search_results`, RRF behavior, retrieval-method mix, searched courses/indexes, and weaknesses.

Internal interfaces:

```python
route_query(query: str, conversation_context: list[dict] | None = None) -> RouterOutput
retrieve(query: str, router_output: RouterOutput | None = None) -> RetrievalRun
metadata_search(query: str, filters: dict | None = None) -> list[RetrievalResult]
merge_with_rrf(result_sets: list[list[RetrievalResult]], k: int = 60, final_top_k: int = 10) -> list[RetrievalResult]
```

Router output:

```text
query_type
candidate_courses
candidate_indexes
keyword_terms
semantic_queries
needs_keyword_search
needs_semantic_search
needs_file_inspection
needs_python
route_confidence
route_reason
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

## Storage and Schema Impact

Read:

- `courses`
- `files`
- `chunks`
- `chunk_fts`
- `embeddings`

Writing `search_runs` and `search_results` belongs to spec 09. This spec should return enough data for that persistence layer.

## Workflow

1. Normalize the query for routing while preserving original text.
2. Match exact course names and common aliases from `courses`.
3. Detect file extensions, logical source-type hints, assignment/project terms, and code/data terms.
4. Select candidate logical indexes.
5. If rule-based routing is ambiguous or empty and LLM fallback is configured, call the LangChain router adapter.
6. Run metadata search for course/file/path matches.
7. Run keyword search when enabled by route.
8. Run semantic search when enabled by route.
9. Merge and deduplicate results with RRF.
10. Return final top K results plus debug coverage fields.
11. Keep `notebooks/retrieval_eda.ipynb` aligned with router fields, retrieval result shape, RRF parameters, and persisted search trace fields.

## Failure and Safety Rules

- If no index has data, return an empty retrieval run with clear weaknesses.
- LLM fallback must be optional and disabled in automated tests.
- If the LLM router returns invalid JSON, fall back to rule-based output and record a weakness.
- Do not execute code even when `needs_python=true`; this flag only informs later safe-inspection decisions.
- RRF is the only MVP merge algorithm. Do not add reranking in this spec.
- The EDA notebook must read generated app data only and must not mutate SQLite, indexes, or `Courses`.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated routing tests for every supported query type.
- Verify exact course-name matching preserves misspellings such as `High Preformance Computing for Big Data`.
- Verify file-finding queries prioritize metadata and keyword search.
- Verify code and data queries select code/data schema indexes.
- Verify ambiguous queries use an injected test-only chat model in isolation from production configuration.
- Verify RRF ranking is deterministic and does not normalize scores from different systems.
- Verify `notebooks/retrieval_eda.ipynb`, once created, is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Optional smoke: run retrieval against a tiny fixture database with keyword and synthetic fixture vector results.

## Implemented runtime contract

Feature 08 requires an explicitly selected reviewed embedding model for every
`retrieve` invocation, including unsupported routes. Supported routes execute
one metadata search, one routed-term keyword search, and one semantic search per
semantic query. Candidate courses and logical indexes are hard filters. Rule
routing is deterministic and falls back to a configured LangChain provider only
when course/index scope or intent remains unresolved; missing configuration,
invalid JSON, or low confidence returns a successful `unknown_or_unsupported`
run without searches, while provider/backend failures are fatal.

The safe result contract extends `RetrievalResult` for file-level metadata
results and adds `RouterOutput`, `RetrievalResultSet`, `RetrievalContribution`,
`FusedRetrievalResult`, and `RetrievalRun`. RRF uses one-based unweighted
`1 / (rrf_k + rank)` contributions, treats each semantic expansion as its own
input list, and retains explicit provenance. Feature 08 does not rerank,
inspect or execute source files, write retrieval tables, or create
`notebooks/retrieval_eda.ipynb`; that notebook waits for Feature 09 traces.

## Acceptance Criteria

- `uv run -m uni_rag_agent retrieve "query"` returns routed courses, indexes, and merged results.
- The router works without an LLM for obvious course/file/index queries.
- LLM fallback is config-driven and testable with an injected test-only chat model.
- Hybrid retrieval uses metadata, keyword, and semantic search where appropriate.
- No reranker is required or invoked for MVP.
- `notebooks/retrieval_eda.ipynb` exists once retrieval traces are persisted and can inspect route/retrieval behavior without mutating generated or source data.
