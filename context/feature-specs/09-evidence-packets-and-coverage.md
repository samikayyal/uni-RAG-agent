# Feature Spec 09: Evidence Packets, Search Persistence, and Coverage

## Purpose

Create the auditable persistence boundary between the mandatory Feature 08
planner/retriever and Feature 10 answering. `retrieve` remains a read-only
diagnostic command; `evidence build` records a validated plan, bounded backend
result sets, the complete RRF ordering, authoritative selected chunks, and the
exact immutable packet passed to the answer generator.

## Depends On

- [08-query-routing-and-hybrid-retrieval.md](08-query-routing-and-hybrid-retrieval.md)
- `context/architecture.md` tables: `search_runs`, `search_result_sets`, `search_results`, `evidence_packets`
- DEC-004, DEC-005, DEC-014, DEC-020, DEC-028, DEC-029, DEC-031, DEC-033, DEC-034, DEC-039

## In Scope

- Persist successful, unsupported, and post-planning failed search lifecycles.
- Store effective non-secret retrieval settings and bounded raw result sets.
- Store the complete deterministic fused RRF candidate ordering.
- Select only authoritative current chunk-backed evidence up to `final_top_k`.
- Enforce the configurable whole-chunk evidence token budget, defaulting to
  `12,000` whitespace-estimated tokens.
- Build canonical, JSON-safe, reloadable evidence packets with structured
  coverage and deterministic weaknesses.
- Preserve the canonical embedding model identity selected through the provider
  registry in retrieval runs, evidence settings, packet projections, and
  telemetry; provider is not supplied through a separate environment variable.
- Add `evidence build` and `evidence show` CLI commands.
- Add the read-only `notebooks/retrieval_eda.ipynb` notebook.

## Out of Scope

- Natural-language answer generation and citation rendering.
- Reranking beyond Feature 08 RRF.
- Source-file inspection, notebook execution, Python execution, Chroma
  mutation, or any mutation under `Courses` during packet assembly.

## Configuration

Add `evidence_max_tokens: int` to `Config` and `Config.as_safe_dict()` with:

```text
UNI_RAG_EVIDENCE_MAX_TOKENS=12000
```

The value must be a positive integer. Blank, non-integer, zero, and negative
values fail configuration validation. Evidence selection uses a persisted
positive `chunks.token_count` when available and otherwise estimates
`len(chunks.text.split())`; it never truncates a selected chunk.

## Public Interfaces

```python
class EvidenceError(RuntimeError): ...

def build_evidence(
    config: Config,
    query: str,
    conversation_context: Sequence[dict[str, str]] | None = None,
    model: str | None = None,
    *,
    chat_model: object | None = None,
) -> EvidenceBuildResult: ...

def load_evidence_packet(
    config: Config,
    evidence_packet_id: int | None = None,
    *,
    search_run_id: int | None = None,
) -> EvidencePacket: ...

def explain_search_coverage(config: Config, search_run_id: int) -> SearchCoverage: ...
```

`load_evidence_packet` requires exactly one positive identifier. A missing
packet reports whether the run is missing, still running, failed, or completed
without a packet. Packet lookup by `search_run_id` is unambiguous because the
schema enforces one packet per run.

Low-level persistence helpers and caller-supplied `QueryPlan` bypasses are
private. Every real build invokes Feature 08 planning exactly once.

## Persisted Models

Evidence-specific immutable models live in a focused retrieval module and
expose recursive `as_safe_dict()` methods. Tuple fields become JSON arrays and
no dataclass, path, secret, or conversation object leaks into persisted JSON.

`RetrievalSettings` contains effective provider/model names, all retrieval
limits, RRF settings, the evidence budget, and the bounded context-message
count. Its `embedding_model` is the canonical registry identifier (for example,
`google/gemini-embedding-001`, never its `gemini-embedding-001` alias). It
contains no API keys, hosted endpoints, or conversation contents.

`EvidenceLocation` contains `type`, `value`, and a deterministic `label`:
`page 12`, `slide 8`, `notebook cell 23`, `function train_model`, or
`location unavailable` when both location fields are null.

`EvidenceItem` contains course, authoritative file/chunk IDs and path, logical
source type, location, complete authoritative chunk text, token count, fused
rank/score, `retrieval_method="hybrid"`, and every RRF contribution.

`SearchCoverage` contains:

```text
search_run_id, status
searched_courses, searched_indexes, keyword_terms, semantic_queries
raw_result_count, raw_result_counts_by_method
fused_candidate_count, selectable_candidate_count, evidence_count
evidence_token_count
courses_with_chunk_hits, indexes_with_chunk_hits, source_types_with_chunk_hits
courses_without_chunk_hits, indexes_without_chunk_hits
semantic_queries_without_hits, missing_capabilities
file_only_candidate_count, token_budget_omission_count
oversized_evidence_omission_count, unselected_selectable_candidate_count
weaknesses
```

Chunk hits are raw keyword/semantic hits, or metadata contributions attached to
such a chunk candidate, that still identify an authoritative chunk. File-only
metadata hits are never chunk hits and never synthetic evidence. Planned lists
preserve validated plan order; derived source types use logical-index order.

`EvidencePacket` contains exactly the query, interpreted intent, complete typed
query plan, settings snapshot, `searched` arrays, coverage, selected evidence,
top-level weaknesses, and these answer constraints in this order:

```text
Answer only from evidence.
Cite course and file.
If evidence is insufficient, say so.
```

Top-level packet weaknesses equal `coverage.weaknesses` exactly.

## Storage and Migration Impact

`search_runs` adds `retrieval_settings_json TEXT NOT NULL DEFAULT '{}'`.
Initialization must migrate a legacy `router_output_json` column to
`query_plan_json` without changing IDs or JSON, add the settings column when
missing, and fail clearly if neither plan column exists. It must be idempotent,
parameterized, and nondestructive.

`search_result_sets` stores one completion envelope for each successful raw
metadata, keyword, or semantic backend call:

```text
search_run_id, result_set_id, retrieval_method, query, result_count, completed_at
```

The envelope is committed atomically with that result set's raw rows, including
when `result_count=0`. It is the authoritative record for partial-run coverage:
an absent envelope means that backend call was not completed, while a zero count
means it completed successfully with no hits.

Add these indexes:

```sql
CREATE INDEX idx_search_results_run_method_rank
ON search_results(search_run_id, retrieval_method, rank);

CREATE UNIQUE INDEX idx_evidence_packets_search_run
ON evidence_packets(search_run_id);
```

Before creating the unique packet index, detect duplicate
`evidence_packets.search_run_id` values and raise `StorageError` without
deleting or merging rows. Preserve `search_results.chunk_id ON DELETE SET NULL`.

Use one canonical serializer everywhere:

```python
json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
```

Construct a packet once, serialize it once, store those exact bytes, and load
with strict required/extra/type validation. Do not reconstruct a stored packet
from current search tables.

## Retrieval and Run Lifecycle

Refactor retrieval behind a private recorder-enabled execution seam while
preserving `retrieve(...) -> RetrievalRun` and its no-write behavior. The seam
must return the public final-top-K view plus the complete fused ordering.

After a plan validates, `evidence build` inserts and commits one `running`
`search_runs` row containing the plan, settings snapshot, searched scopes, and
context count. Planning failures create no run.

For each successful backend call, commit one complete raw result set. Raw rows
use `metadata`, `keyword`, or `semantic`, retain native rank/score, and store a
canonical result-set/query envelope. Duplicate chunks across result sets remain
separate rows. After all backends succeed, persist the full fused ordering as
`hybrid` rows with complete contribution provenance. Raw and fused rows start
with `selected_for_evidence=0`.

Valid `unknown_or_unsupported` plans create no backend rows, a zero-evidence
packet, and finish with status `unsupported`. Supported zero-hit runs finish
with status `completed` and a zero-evidence packet.

Backend failures after planning finish the run as `failed`, retain all complete
result sets committed before the failure, store only a bounded sanitized domain
error, create no fused rows unless fusion already completed, and create no
packet. Packet-assembly or authoritative-drift failures roll back selection and
packet insertion, then mark the run failed in a separate transaction while
retaining raw/fused audit rows.

The semantic backend inherits Feature 07's provider contract: local profiles use
the `embeddings` extra, hosted Google/Nebius profiles use
`embeddings-cloud`, hosted vectors use declared dimensions with actual-response
validation and no dedicated probe, and embedding work uses shared three-total-
attempt retry rules for network/408/429/5xx failures, 64-chunk batching, and
per-batch commits. If hosted retries are exhausted,
already committed index batches remain durable and a later incremental index run
resumes missing chunks. Missing-extra, missing-credential, and provider errors
are sanitized before they reach run records, packets, telemetry, or CLI output.

Hosted evidence workflows send eligible course text and semantic queries to an
external provider and may incur charges. Local profiles keep model execution
local apart from model downloads as applicable. Manual credentialed hosted smokes
are optional and are never required by automated packet tests.

## Evidence Selection and Authoritative Safety

Selection walks the complete fused order. File-only candidates count toward
coverage but never consume the evidence count. For each chunk candidate, a
write transaction hydrates the exact `chunks.id` + `chunks.file_id` join and
requires the current authoritative file to be indexed, present in the latest
inventory, non-metadata-only, and paired with nonblank supported-source text.
Candidate course/path/source/location identity must still match. Any required
drift fails the entire build; snippets are never promoted to evidence.

Positive authoritative token counts are used as-is; invalid counts use the
whitespace estimator. Oversized whole chunks are skipped and counted. A chunk
that would overflow the remaining budget is omitted, but lower-ranked smaller
chunks may still backfill. Evidence never exceeds `final_top_k` or the token
budget.

## Coverage and Weaknesses

Weaknesses are deterministic, exact-deduplicated, and ordered as follows:

1. Existing Feature 08 retrieval weaknesses.
2. The planner reason for `unknown_or_unsupported`.
3. Missing `file_inspection` and/or `python_execution` capabilities.
4. Planned courses without chunk hits.
5. Planned indexes without chunk hits.
6. Empty semantic result sets using their exact planned query strings.
7. Query-relevant matched ineligible-file limitations grouped by category/reason.
8. Token-budget and oversized-item omission summaries.
9. A final insufficient-evidence warning when no evidence is selected.

Source-type limitations are query-relevant only; unrelated packets do not get
global archive boilerplate. `explain_search_coverage` reports partial counts
for failed runs without pretending a packet exists.

## CLI and Notebook

Add:

```powershell
uv run -m uni_rag_agent evidence build "Explain MapReduce" --model BAAI/bge-m3
uv run -m uni_rag_agent evidence build "Explain MapReduce" --model BAAI/bge-m3 --debug
uv run -m uni_rag_agent evidence build "Explain MapReduce" --model BAAI/bge-m3 --json
uv run -m uni_rag_agent evidence show --search-run-id 1
uv run -m uni_rag_agent evidence show --search-run-id 1 --json
```

`--json` build prints exactly one complete build-result object. Text/debug
output includes both IDs, scopes, raw/fused/selectable/evidence counts, token
total, selected rows, and weaknesses. Use `EVIDENCE_ERROR = 8` for packet
assembly/loading/validation errors; retain `SEARCH_ERROR = 7` for planner and
retrieval failures and `STORAGE_ERROR = 3` for schema/readiness failures.
The `--model` value may be a local reviewed profile or either hosted canonical
profile/accepted Gemini alias; provider construction is inferred from the
registry and the matching embedding extra must be installed. Credentialed
hosted packet smokes are optional.

Create `notebooks/retrieval_eda.ipynb` with pandas and matplotlib only. It must
resolve the repository root from either the repo root or `notebooks/`, open
SQLite with `mode=ro` and `PRAGMA query_only = ON`, document the read-only
boundary, guard empty/malformed historical JSON, and inspect runs, planning
confidence, settings, raw/fused results, contributions, selected evidence,
coverage, weaknesses, token budgets, unsupported/zero-evidence runs, and
partial failed runs. Clear outputs and execution counts.

## Tests and Acceptance Criteria

Add focused tests for configuration, schema migrations/uniqueness, strict model
serialization, success/unsupported/zero-hit/partial-failure lifecycles,
authoritative drift, token selection, coverage/weakness ordering, CLI output and
exit codes, read-only `retrieve`, and notebook safety. Run the focused suite and
the full suite with `uv`, then verify help output, schema state, and `git diff
--check`.

Acceptance requires that every successful or unsupported build has exactly one
immutable packet; raw and complete fused rows are auditable; only selected
authoritative current chunks become evidence; conversation contents and source
files never enter persistence; and `retrieve` remains fully read-only.
