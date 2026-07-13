# Feature 08: Mandatory LLM Query Planning and Hybrid Retrieval

## Purpose

Implement the read-only retrieval slice. Every `retrieve` invocation must first
call a configured LLM exactly once to produce a validated structured `QueryPlan`.
The deterministic application then performs metadata, keyword, and semantic
retrieval under the approved hard filters and merges results with RRF.

Read first:

- `context/project_overview.md`
- `context/architecture.md`
- `context/decisions.md` (especially DEC-005, DEC-014, DEC-031, DEC-033)
- Features 02, 06, and 07

## In Scope

- Mandatory LLM query planning through LangChain with the configured provider/model.
- Canonical course names loaded from SQLite, bounded validated conversation context,
  exact JSON validation, and course-name casing normalization.
- Supported query types: `concept_explanation`, `course_summary`,
  `cross_course_comparison`, `find_file`, `assignment_or_project_lookup`,
  `code_question`, `data_question`, `study_quiz`, `portfolio_resume`, and
  `unknown_or_unsupported`.
- One metadata search, one keyword search, and one semantic search for each
  planned semantic query for every supported plan.
- Existing hard course/index filters, RRF-only fusion, provenance, safe CLI/JSON
  output, telemetry, and strict reviewed embedding-model precondition.
- Offline tests through injected chat models; no production fake provider.

## Out of Scope

- Rule-based routing, aliases, cue matching, extension-driven planning,
  stopword term extraction, or fuzzy course matching.
- Dynamic tool calls, reranking, source-file inspection/execution, persistence,
  evidence packets, answer generation, and UI work.

## Public Interfaces

```python
class QueryPlanningError(RuntimeError): ...

@dataclass(frozen=True)
class QueryPlan:
    query_type: str
    candidate_courses: tuple[str, ...]
    candidate_indexes: tuple[str, ...]
    keyword_terms: tuple[str, ...]
    semantic_queries: tuple[str, ...]
    needs_file_inspection: bool
    needs_python: bool
    plan_confidence: float
    plan_reason: str

plan_query(config, query, conversation_context=None, *, chat_model=None) -> QueryPlan
retrieve(config, query, conversation_context=None, model=None, *, chat_model=None) -> RetrievalRun
```

`RetrievalRun.query_plan` serializes as `query_plan`; its plan fields serialize
as `plan_confidence` and `plan_reason`. No router-named API, environment
variable, telemetry field, compatibility alias, or caller-supplied plan bypass
exists.

## Query-Planning Contract

1. Normalize a nonblank query, load canonical SQLite course names, validate at
   most six current context messages, and build a prompt containing the query,
   courses, allowed logical indexes, query types, semantic-query limit, and
   exact output schema.
2. Require `UNI_RAG_LLM_PROVIDER` and `UNI_RAG_LLM_MODEL`; construct exactly
   that provider/model unless the test chat-model seam is injected.
3. Invoke once and parse one JSON object with exactly the `QueryPlan` fields.
4. A supported plan requires a supported non-unknown type, one or more known
   canonical courses and logical indexes, nonblank keyword terms, one through
   `semantic_query_limit` nonblank semantic queries, boolean flags, a nonblank
   reason, and confidence at or above `query_plan_min_confidence` (0.60).
5. `unknown_or_unsupported` requires empty course/index/keyword/semantic scope;
   it produces a successful empty run, skips all backends, and records its LLM
   reason as a weakness.
6. Missing configuration, unavailable optional dependency, provider construction
   or invocation failures, invalid JSON/schema/values, unknown types/courses/
   indexes, excessive terms/queries, and low confidence raise
   `QueryPlanningError` and fail retrieval.

## Configuration and CLI

- `query_plan_min_confidence` comes from
  `UNI_RAG_QUERY_PLAN_MIN_CONFIDENCE` and defaults to `0.60`.
- `filename_fuzzy_threshold`, `path_fuzzy_threshold`, and RapidFuzz remain for
  deterministic metadata retrieval. There is no course fuzzy threshold.
- LLM settings remain nullable during global configuration loading so Features
  01–07 and direct search commands run without the optional `llm` extra.
  `retrieve` requires the LLM pair and `uv sync --extra llm` as well as an
  explicit reviewed embedding model.
- Planning telemetry uses `query_planning_completed` or
  `query_planning_unsupported` and includes query type, plan confidence, planned
  courses/indexes, semantic-query count, and configured provider/model without
  credentials or conversation contents.

## Failure and Safety Rules

- The feature is read-only: do not write search runs/results, evidence packets,
  SQLite records, Chroma records, source files, or course files.
- `retrieve` remains read-only after Feature 09: the persisted `evidence build`
  workflow uses a private recorder-enabled execution seam around the same
  planner/backend/RRF sequence and does not expose a caller-supplied plan bypass.
- Keep all backend failures fatal; zero results remain coverage weaknesses.
- Apply planned course and index scopes as hard filters before result limits.
- Keep DEC-014's one-based unweighted RRF and no reranker.

## Tests and Acceptance Criteria

- Cover valid supported, multi-course/index, canonicalization, term/query
  deduplication and limits, bounded context, and valid unsupported plans.
- Cover missing configuration, provider failure, invalid JSON/non-object,
  missing/extra fields, unknown values, blank/incorrect fields, and confidence
  boundaries.
- Prove `retrieve` always obtains its plan through the planner, has no prebuilt
  plan bypass, runs all three methods for supported plans, runs no backend for a
  valid unsupported plan, and performs no Feature 09 persistence.
- Verify config/CLI field and telemetry names, safe logs, exit codes, and no
  router/fallback terminology in live help.
- Run focused planner/retrieval/config/CLI tests, package compilation,
  `uv run pytest -q`, and `git diff --check`.
