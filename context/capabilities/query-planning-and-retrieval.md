# Query planning and retrieval

## Current behavior

`plan_query()` calls the configured chat model exactly once and validates a
structured `QueryPlan`: query type, canonical course/index scopes, keyword
terms, semantic queries, inspection flags, confidence, and reason. Logical
index validation uses the taxonomy derived from `search_contracts.py`. A valid
`unknown_or_unsupported` plan intentionally runs no backends. For supported
plans, `retrieve()` executes deterministic metadata, FTS5 keyword, and Chroma
semantic searches with hard planned filters and merges their provenance using
unweighted Reciprocal Rank Fusion. Multi-word planner keyword terms expand to
deduplicated token-level OR matches rather than strict FTS phrases. All planned
semantic queries pass through one `semantic_search_many()` request context,
while each query still produces its own ordered result set and audit identity.
Zero hits become coverage
weaknesses; a backend/provider failure is fatal. Metadata may return file-level
rows with no chunk id, but cannot become evidence text by itself.

`retrieve` is non-persisting with respect to SQLite search/evidence rows, Chroma,
and `Courses/` source files; the CLI still writes JSONL run telemetry under
`data/runs/`. `evidence build` is the persistence caller described in
[evidence-and-answering.md](evidence-and-answering.md).

## Public entry points

- `uv run -m uni_rag_agent retrieve "query" --model <profile> [--debug] [--json]`
- Python: `plan_query(config, query, conversation_context=None, *, chat_model=None)`
  and `retrieve(config, query, conversation_context=None, model=None, *, chat_model=None)`.
- Models: `QueryPlan`, `RetrievalResult`, `RetrievalResultSet`, `RetrievalRun`,
  `FusedRetrievalResult`, and `RetrievalError` in `retrieval/models.py` and
  `retrieval/core.py`.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/retrieval/{planner,metadata,rrf,core,models}.py`.
- Tests: `tests/test_query_planning.py`, `tests/test_hybrid_retrieval.py`,
  plus CLI/config coverage in `tests/test_cli.py` and `tests/test_config.py`.
- Generated: safe CLI result JSON and retrieval JSONL telemetry under
  `data/runs/`; search/evidence persistence artifacts are created only by
  `evidence build`.

## Invariants and failure boundaries

- Planner configuration and the `llm` extra are required only when retrieval is
  executed; global `config check` may succeed with nullable model settings.
- Planner output must be one JSON object satisfying the schema, canonical course
  names, known logical indexes, and supported query type. Malformed output and
  provider failures raise `QueryPlanningError`. A plan whose confidence is below
  the configured threshold does not raise: it is downgraded to an
  `unknown_or_unsupported` plan with empty scopes and an explanatory
  `plan_reason`, so the ask pipeline returns an honest insufficient-evidence
  answer instead of a provider error.
- Planned courses/indexes are hard filters, with one deliberate broadening: a
  plan scoping to `slides_index` without `document_index` gets `document_index`
  appended, because slide decks are frequently ingested with
  `source_type=document` (e.g. PDF exports) and a slides-only scope would
  silently exclude them. RRF preserves backend method,
  source rank, native score, semantic-query identity, and contribution fields;
  no reranker or score normalization is inserted.
- One retrieval request constructs one embedding provider and Chroma client,
  reuses collection handles across all semantic queries, and preserves exactly
  one result set per semantic query in planner order.
- Errors are sanitized and do not expose keys, prompts, or provider response
  bodies. Unsupported plans are successful empty results with a reason.

Binding decisions: [DEC-014/033](../decisions.md#dec-014033--mandatory-planner-deterministic-hybrid-retrieval-rrf),
[DEC-011](../decisions.md#dec-011--sqlite-authority-with-chroma-logical-indexes),
and [DEC-023/029/028/040](../decisions.md#dec-023029028040--current-file-and-deletion-semantics).
