# Glossary

These terms describe this project’s domain and user-visible contracts.

- **Answer trace** — The append-only record of an answer produced from one
  evidence packet, including rendered text, structured citations, limitations,
  and safe model identity.
- **Coverage** — A compact account of what a query searched, which sources
  contributed, what was found, and what remains missing or weak.
- **Current file** — A source file whose latest inventory state is eligible for
  normal extraction/indexing and retrieval; historical or missing files are not
  current.
- **Evidence item** — One authoritative, source-located chunk selected for a
  packet, retaining its course, path, location, text, rank, and provenance.
- **Evidence packet** — The immutable, auditable handoff from retrieval to
  answering: selected evidence plus the query interpretation and coverage.
- **Fixture state** — Isolated generated inventory, extraction, keyword, and
  vector data used by the committed evaluation fixtures rather than the normal
  archive state.
- **Logical index** — A stable retrieval scope such as `slides_index` or
  `code_index`, corresponding to one chunk source type.
- **Query plan** — The validated interpretation of a user query: intent,
  canonical course/index scope, search terms, semantic queries, and confidence.
- **Retrieval run** — The result of planning and backend search for one query,
  whether returned without search/evidence persistence or recorded by an
  evidence build.
- **Result set** — Results from one retrieval backend or semantic query before
  fusion, with a completion status and source method.
- **Source location** — The human-readable position supporting evidence, such as
  a page, slide, notebook cell, schema section, code symbol, or timestamp.
- **Source admission** — The decision at inventory time that determines whether
  a file is extractable/indexable or metadata-only.
- **Coverage weakness** — A deterministic explanation for missing, empty, or
  omitted support, such as a zero-hit backend or a token-budget exclusion.
- **Canonical embedding profile** — A reviewed embedding model identity after
  alias resolution; it determines provider, vector dimension, and physical
  collection identity.
