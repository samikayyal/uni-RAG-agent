# Deployment reference

This page records the completed public deployment and its operating contract. It
is a reference for maintaining the service, not a provisioning tutorial. The
application never creates or modifies Google Cloud or Cloudflare resources;
those remain operator-owned.

**Last control-plane and endpoint verification:** 2026-07-26. Cloud Run reported
`uni-rag-agent-00005-v2q` as the serving revision with 100% traffic; the custom
domain mapping and certificate were ready, its DNS resolved to Google's target,
and `https://unirag.samikayyal.com/ready` returned a fully ready response.
Revision, image, secret-version, billing, and DNS state are operational data and
must be rechecked before a change.

## Public endpoints and resources

| Concern | Current value | Notes |
| --- | --- | --- |
| Google Cloud project | `uni-rag-agent` | Dedicated billed project. |
| Region | `europe-west1` | Cloud Run, Artifact Registry, and Firestore location. |
| Cloud Run service | `uni-rag-agent` | Generation 2 public service. |
| Generated Cloud Run URL | `https://uni-rag-agent-lowhir3jjq-ew.a.run.app` | Keep available as a rollback endpoint. |
| Public custom domain | `https://unirag.samikayyal.com` | Google Cloud Run domain mapping; see [Cloudflare and DNS](#cloudflare-and-dns). |
| Serving revision at last verification | `uni-rag-agent-00005-v2q` | 100% traffic at the time stated above. |
| Artifact Registry repository | `europe-west1-docker.pkg.dev/uni-rag-agent/uni-rag` | Release images use immutable tags/digests, never `latest`. |
| Runtime service account | `uni-rag-runtime@uni-rag-agent.iam.gserviceaccount.com` | Attached to the Cloud Run revision. |
| Firestore database | `(default)` Native mode | Holds public-demo abuse controls and the durable authenticated-ask ledger. |

Cloud Run is intentionally the public ingress. The service is unauthenticated
at the platform layer because a browser demo must reach it, but application
access is protected by Cloudflare Turnstile, a signed client-bound demo token,
and server-side quotas.

## Image and runtime topology

The container is a CPU-only, non-root Python 3.12 image. The Dockerfile builds
the project with the `embeddings`, `embeddings-cloud`, `llm`, and
`public-demo` extras, then copies only the runtime virtual environment into the
final image.

Three operator-selected inputs are baked into every release image:

| Image input | Container location | Purpose |
| --- | --- | --- |
| `deployment/assets/models/embeddinggemma-300m` | `/app/models/embeddinggemma-300m` | Offline EmbeddingGemma snapshot, retained for on-demand local-profile selection. |
| `data/uni_rag.sqlite` | `/app/seed-data/uni_rag.sqlite` then `/data/uni_rag.sqlite` | Selected archive database. The entrypoint copies the seed to mutable runtime state at each boot. |
| `data/indexes/vector/` | `/data/indexes/vector` | Chroma vector collections for the selected archive. |

`/app` contains packaged application assets. `/data` contains mutable runtime
state: the copied SQLite database, vectors, extraction directory, and run-log
directory. The image does not include `Courses/`, arbitrary `data/` contents,
or deployment-input validation. Selecting the database, vector collections, and
accepted model snapshot is an operator responsibility.

The entrypoint starts Uvicorn on Cloud Run's `PORT` (normally `8080`) with
forwarded headers enabled. The image must retain `/app/.venv/bin` on `PATH`;
a revision that cannot find `uvicorn` exits before readiness can succeed.

## Cloud Run service configuration

| Setting | Current value | Reason |
| --- | --- | --- |
| Port | `8080` | Uvicorn listener and both HTTP probes. |
| CPU / memory | 2 vCPU / 8 GiB | Supports the optional local embedding model when selected. |
| Concurrency | 4 | Deliberately above the two expensive ask slots so UI/control requests remain responsive. |
| Minimum / maximum instances | 0 / 1 | One bounded public-demo instance; cold starts are expected. |
| Request timeout | 300 seconds | Platform headroom around the application ask timeout. |
| CPU allocation | Request-based | CPU is throttled outside requests. |
| Startup CPU boost | Enabled | Retained Cloud Run startup configuration. |
| Execution environment | Second generation | Current Cloud Run revision setting. |
| Startup probe | `GET /ready`, port 8080, 10-second timeout and period, threshold 24 | Allows up to four minutes for startup. |
| Readiness probe | `GET /ready`, port 8080, 5-second timeout and period, threshold 3 | Controls serving readiness. |

`GET /health` is liveness-only. `GET /ready` additionally verifies storage and
the serving default's vector index; it neither constructs EmbeddingGemma nor
calls Gemini or Nebius. Optional profiles do not gate readiness.
The last public readiness response was:

```json
{"status":"ready","storage_ready":true,"default_vector_index_ready":true}
```

For a failed revision, inspect its Cloud Logging stderr and startup events before
changing probes. Probe failure can be caused by an early process exit, missing
credentials, storage, or model startup—not only slow readiness.

## Runtime configuration and secrets

The public-demo switch is enabled (`UNI_RAG_PUBLIC_DEMO_ENABLED=true`). The
deployed planner and answer models are Gemini, while the selected default
embedding profile is Nebius:

| Configuration | Current value |
| --- | --- |
| Planner | `gemini` / `gemini-3-flash-preview` |
| Answerer | `gemini` / `gemini-3.5-flash` |
| Default embedding profile | `Qwen/Qwen3-Embedding-8B` |
| Firestore project/database | `uni-rag-agent` / `(default)` |
| Maximum public query length | 4,000 characters |
| Ask capacity / capacity wait | 2 / 10 seconds |
| Minute / client-day / global-day limits | 3 / 10 / 100 |
| Demo token TTL | 1,800 seconds (30 minutes) |
| Application ask timeout | 180 seconds |

Secret values never appear in the repository or ordinary Cloud Run
environment-variable values. The currently referenced Secret Manager bindings
all use explicit numeric version `1`:

| Environment variable | Secret Manager secret | Role |
| --- | --- | --- |
| `GOOGLE_API_KEY` | `google-api-key:1` | Primary Gemini planner and answer credential. |
| `GOOGLE_API_KEY_2` | `google-api-key-2:1` | Gemini quota failover for planner and answer calls. |
| `GOOGLE_API_KEY_EMBEDDING` | `google-api-key-embedding:1` | Dedicated credential for `google/gemini-embedding-001`; it never falls back to a chat key. |
| `NEBIUS_API_KEY` | `nebius-api-key:1` | Nebius embedding profile credential. |
| `UNI_RAG_DEMO_TOKEN_SIGNING_SECRET` | `demo-token-signing-secret:1` | Signs client-bound demo tokens. |
| `TURNSTILE_SECRET_KEY` | `turnstile-secret-key:1` | Server-side Turnstile verification. |

Planner and answer calls start on `GOOGLE_API_KEY`. On a quota/resource-exhausted
error, they rotate to `GOOGLE_API_KEY_2`, retain that selected key until the
next such error, then wrap around; one invocation tries each configured key at
most once. This never applies to Gemini embeddings, which use only
`GOOGLE_API_KEY_EMBEDDING`.

The runtime account has `roles/datastore.user` for Firestore and per-secret
`roles/secretmanager.secretAccessor` grants. The Cloud Build identity receives
Artifact Registry writer on the repository; neither identity should be widened.

## Public-demo protection and Firestore

Cloudflare Turnstile is completed before a public demo session is created. A
valid, single-use response becomes a signed, client-bound, per-tab token.
Browser state and completed response payloads stay in that tab's
`sessionStorage`; a token for one client/session cannot operate another
session's ask, progress, cancellation, or liveness routes.

Firestore is the cross-instance enforcement store for quota counters,
client-digest UTC buckets, minute buckets, and idempotent request reservations.
It also retains every authenticated ask attempt in `demo_asks`, including the
raw query and client IP, requested/effective safe settings, model identities,
outcome/timing metadata, sanitized errors, and the complete safe response with
evidence and answer content. These records have no expiry. Tokens, Turnstile
responses, authorization headers, secrets, and raw exceptions remain excluded.
The quota collections are `demo_quota_global`, `demo_quota_clients`,
`demo_quota_minutes`, and `demo_quota_reservations`.

Quota is reserved atomically only after an ask obtains one of the two in-process
ask slots. Invalid, unauthorized, over-limit, and capacity-rejected requests do
not consume quota; accepted provider failures and cancellations do. A completed
request-ID replay is rejected before it starts provider work. Turnstile or
Firestore infrastructure failure produces the safe 503
`abuse_service_unavailable` response.

### Operator steps for the ask ledger

Application code never changes Firestore indexes or IAM. Before deploying the
first image that writes `demo_asks`, the operator must exempt its large/nested
fields from automatic indexing. Run these commands from an authenticated local
PowerShell session; none are executed by the application:

```powershell
$ProjectId = "uni-rag-agent"
$Database = "(default)"
$Collection = "demo_asks"
$UnindexedFields = @(
  "query",
  "requested_retrieval_settings",
  "effective_settings",
  "models",
  "client",
  "quota_remaining",
  "trace_ids",
  "timing",
  "error",
  "response"
)

foreach ($Field in $UnindexedFields) {
  gcloud.cmd firestore indexes fields update $Field `
    --project=$ProjectId `
    --database=$Database `
    --collection-group=$Collection `
    --disable-indexes
}
```

The same intended exemptions are recorded in
[`deployment/firestore.ask-audit-indexes.json`](../deployment/firestore.ask-audit-indexes.json)
for review. Keep the ordinary automatic index on `received_at`; the local
dashboard orders and paginates by that field. Verify the applied exemptions:

```powershell
gcloud.cmd firestore indexes fields list `
  --project="uni-rag-agent" `
  --database="(default)" `
  --filter='collectionGroup:demo_asks'
```

No new runtime IAM role is required: the existing
`roles/datastore.user` grant lets the runtime create/update audit documents.
The collection is created by the first authenticated submission after the new
revision is serving.

For local dashboard access, authenticate Application Default Credentials and
install both optional dependency groups:

```powershell
gcloud.cmd auth application-default login
uv sync --extra public-demo --extra dashboard
```

Download the MaxMind GeoLite2 City `.mmdb` database after accepting its license,
place it at `data/GeoLite2-City.mmdb`, and keep it out of Git. Then set the
Firestore project in `.env` and start the loopback-only viewer:

```dotenv
UNI_RAG_FIRESTORE_PROJECT_ID=uni-rag-agent
UNI_RAG_FIRESTORE_DATABASE=(default)
```

```powershell
uv run -m uni_rag_agent app audit-dashboard
```

Open `http://127.0.0.1:8001`. Use `--geoip-db <path>` for a different database
location and `--port <port>` for a different loopback port. GeoLite2 locations
are approximate analytics signals, not household/address identification.

## Cloudflare and DNS

Cloudflare remains the authoritative DNS provider for `samikayyal.com`; it is
not an HTTP proxy for this service.

| Item | Completed configuration |
| --- | --- |
| Cloud Run mapping | `unirag.samikayyal.com` → `uni-rag-agent` in `europe-west1` |
| Google mapping status | `Ready`, `CertificateProvisioned`, and `DomainRoutable` were true at last verification. |
| DNS record | `CNAME unirag` → `ghs.googlehosted.com.` |
| Cloudflare proxy | **DNS only** (gray cloud), TTL Auto. |
| TLS | Google-managed certificate on the Cloud Run domain mapping. |
| Domain ownership | The `samikayyal.com` Google Search Console verification TXT record is retained. |
| Turnstile widget | One managed widget authorizes both the generated `run.app` hostname and `unirag.samikayyal.com`. |

DNS-only is deliberate. Cloudflare's proxied origin-read limit leaves too little
margin above the application's expensive ask duration. Do not orange-cloud this
hostname, enable Cloudflare redirect behaviour during Google certificate
validation, point DNS at the generated `run.app` hostname, or replace the
Google-provided CNAME with a guessed value. Cloud Run domain mappings are a
Preview feature suitable for this bounded demo; a production-critical service
should move to a Google external Application Load Balancer.

The generated `run.app` hostname stays authorized in Turnstile and is the
immediate fallback if custom-domain DNS or certificate behaviour regresses.

## Ongoing operations

Each release is a newly built immutable Artifact Registry image based on the
three selected image inputs. Verify a revision privately before assigning public
traffic to that same image digest; do not make an unauthenticated revision the
first runtime test of a new archive, vector index, model snapshot, or
configuration.

This checkout uses the tracked `.githooks/post-commit` hook (enabled through the
repository-local `core.hooksPath`) to queue `deployment/submit-build.ps1` after
each successful commit. The commit returns immediately while a background
worker uploads the complete local context, including ignored deployment inputs.
The worker implements latest-request-wins behavior: it serializes local uploads,
cancels ongoing builds for this exact Artifact Registry image before submitting
an immutable timestamp/commit-tagged replacement, and cancels its own new build
if another commit arrived during upload. Run
`.\deployment\submit-build.ps1` for the same explicit shortcut without a commit,
or `.\deployment\submit-build.ps1 -DryRun` to validate inputs and preview the
action. Per-request logs are local under `.git/uni-rag-build/`.
Set `UNI_RAG_SKIP_BUILD=1` in the committing process for a deliberate one-commit
skip.

Rollback is a Cloud Run traffic change to a known-good revision, followed by
verification of active traffic allocation, logs, and `/ready`. The generated
Cloud Run URL is the domain-level fallback; removing a domain mapping is a
DNS/control-plane action, not an application redeployment.

Secret rotation creates a new numeric version, deploys a revision that
references that exact version, validates it, then disables the old version only
after rollback is no longer needed. Rotating the signing secret immediately
invalidates active demo tokens. Quota documents normally expire by their UTC
minute/day scope; manual Firestore deletion must be restricted to confirmed test
documents, never an entire quota collection. `demo_asks` records intentionally
do not expire; deletion is a deliberate operator action.

Use Cloud Run's revision-specific logs and metrics for startup, request-error,
latency, and instance-count investigation. Before attributing a failure to the
probe, inspect process logs. For custom-host changes, recheck the Cloud Run
mapping condition, exact DNS records, HTTPS `/ready`, and a fresh browser
Turnstile/ask flow; browser session state is origin-scoped.

Budget alerts are warnings, not a spending cap. Provider-side Gemini and Nebius
limits remain part of operational cost control.

## Related project contracts

- [Web application capability](capabilities/web-application.md) defines the
  public/local runtime boundary and API behaviour.
- [Configuration and storage capability](capabilities/configuration-and-storage.md)
  defines environment-variable validation and the safe configuration surface.
- [DEC-042](decisions.md#dec-042--explicit-public-demo-boundary-and-deployment-contract)
  is the binding decision for the public-demo/deployment boundary.
- [Operations](operations.md) describes local generated-state preparation.
- `deployment/GCP_RUNBOOK.md` remains a local, ignored historical provisioning
  guide; this page is the tracked deployment reference.
