# Evidence and answering

## Current behavior

`build_evidence()` wraps the planner/retriever with a persistence recorder. It
stores the validated plan/settings, each complete raw result set (including
successful empty sets), fused rows, run status, deterministic coverage, and one
canonical packet. Packet selection uses authoritative current chunks only,
whole chunks in fused rank order, `final_top_k`, and the positive
`UNI_RAG_EVIDENCE_MAX_TOKENS` budget (12,000 by default). File-only metadata
rows remain coverage/audit records. `load_evidence_packet()` and
`explain_search_coverage()` read persisted state without re-running search.

`generate_answer()` receives only a packet and separate answer-model settings.
The model must return one JSON object with `answer_paragraphs` and
`limitations`. Application validation maps packet positions to stable `[E1]`
markers and references, rejects citation/lookalike prose, and reports omitted
evidence as limitations. Empty evidence or a budget that fits no item yields a
deterministic no-provider answer. `store_answer()` reloads and validates the
packet before appending to `answers`; prompts, conversations, keys, and invalid
raw output are never persisted. `AnswerSession` keeps bounded complete turns
for the planner only. `answer_body()` owns the rendered prose/tail boundary,
and `answer_status()` owns deterministic outcome classification for both fresh
and rehydrated answers. `ask` composes build plus answer.

## Public entry points

- `uv run -m uni_rag_agent evidence build "query" --model <profile> [--debug] [--json]`
- `uv run -m uni_rag_agent evidence show --search-run-id <id> [--json]`
- `uv run -m uni_rag_agent answer --evidence-packet-id <id> [--json]`
- `uv run -m uni_rag_agent ask "query" --model <profile> [--json]`
- Python: `build_evidence`, `load_evidence_packet`,
  `explain_search_coverage`, `generate_answer`, `store_answer`, `load_answer`,
  `answer_body`, `answer_status`, and `AnswerSession`.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/retrieval/{evidence,evidence_persistence,evidence_models}.py`
  and `src/uni_rag_agent/answering/{core,persistence,session,audit,providers}.py`.
- Tests: `tests/test_evidence_packets.py`, `tests/test_answering.py`; CLI
  lifecycle assertions live in `tests/test_cli.py`.
- SQLite: `search_runs`, `search_result_sets`, `search_results`,
  `evidence_packets`, and append-only `answers`. Read-only EDA companions are
  `notebooks/retrieval_eda.ipynb` and `notebooks/answering_eda.ipynb`.

## Invariants and failure boundaries

- `retrieve` remains non-persisting with respect to SQLite search/evidence rows,
  Chroma, and `Courses/` source files; the CLI still writes JSONL run telemetry
  under `data/runs/`. Planning failure creates no run; a backend failure after
  planning preserves committed partial audit rows as failed; drift or packet
  assembly failure creates no packet and selects no rows.
- Each successful or valid unsupported run has at most one packet. Packet
  evidence must match current file/course/path/source/location and nonblank
  chunk text; packet JSON is immutable after creation.
- Evidence items, citations, and references carry course-relative file paths
  (`files.relative_path`); absolute host paths never enter packets, prompts, or
  rendered answers. Packets persisted before this change retain their original
  absolute paths.
- Answer citations resolve to packet-authoritative positions or the explicit
  `chunk:<id>` compatibility alias and are revalidated at the append-only
  boundary. Invalid model output retries according to
  `UNI_RAG_ANSWER_MAX_RETRIES`; provider construction/invocation failure creates
  no answer row.
- `ask` may leave a stored packet when answer generation fails or times out;
  the independent answer failure is surfaced without deleting evidence.

Binding decisions: [DEC-034](../decisions.md#dec-034--persisted-evidence-boundary),
[DEC-035/020](../decisions.md#dec-035020--strict-packet-only-answers-and-citations),
and [DEC-014/033](../decisions.md#dec-014033--mandatory-planner-deterministic-hybrid-retrieval-rrf).
