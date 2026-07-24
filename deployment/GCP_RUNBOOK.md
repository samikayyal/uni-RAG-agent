# Cloud Run deployment runbook

This runbook covers the operator-owned GCP and Cloudflare work. Run commands in
PowerShell from the repository root. The application code never creates cloud
resources. Use a dedicated billed project and `europe-west1` throughout.

Official references: [Cloud Run deploy flags](https://docs.cloud.google.com/sdk/gcloud/reference/run/deploy),
[Cloud Run console deployment](https://docs.cloud.google.com/run/docs/deploying),
[Artifact Registry repositories](https://cloud.google.com/artifact-registry/docs/repositories/create-repos),
[Cloud Build container builds](https://docs.cloud.google.com/build/docs/building/build-containers),
[Firestore database creation](https://cloud.google.com/firestore/docs/manage-databases),
[service accounts](https://docs.cloud.google.com/iam/docs/service-accounts-create),
[Secret Manager access](https://docs.cloud.google.com/secret-manager/docs/manage-access-to-secrets),
[billing budgets](https://docs.cloud.google.com/billing/docs/how-to/budgets),
and [Turnstile hostname management](https://developers.cloudflare.com/turnstile/additional-configuration/hostname-management/).

The **Google Cloud Console** instructions below are the website equivalent of the
CLI commands. Console labels can move slightly as Google updates the UI; use the
named product page and the exact values in this runbook. Values such as the
project ID, service account email, image URI, service URL, and Turnstile site key
are created during the preceding steps and must be carried forward.

## 1. Prepare image inputs

The application does not prepare, inspect, or scan deployment inputs. Use the
generated database and indexes you intend to deploy and an already accepted
local EmbeddingGemma snapshot. The Docker and Cloud Build contexts include only
`data/uni_rag.sqlite` and `data/indexes/vector/` from the repository's generated
state, so only the external model snapshot needs to be staged:

```powershell
$ModelSource = 'D:\path\to\hub\models--google--embeddinggemma-300m\snapshots\FULL_REVISION'
New-Item -ItemType Directory -Force deployment/assets/models | Out-Null
Copy-Item $ModelSource deployment/assets/models/embeddinggemma-300m -Recurse
```

Secret review, history cleanup, model acceptance, vector population, and any
other pre-deployment checks are your responsibility. The Docker build only
requires that staged model plus `data/uni_rag.sqlite` and
`data/indexes/vector/`.

### Google Cloud Console

There is no GCP Console replacement for this local preparation step. The inputs
are read from the Docker build context, so stage the model and review all three
inputs locally before starting the Cloud Build or Cloud Run workflow. Do not
upload `Courses/` or other unrequested generated state.

## 2. Select and bill the deployment project

Use the existing dedicated project ID `uni-rag-agent`:

```powershell
$ProjectId = 'uni-rag-agent'
$Region = 'europe-west1'
$Repository = 'uni-rag'
$Service = 'uni-rag-agent'
$RuntimeAccount = 'uni-rag-runtime'

gcloud config set project $ProjectId
gcloud billing accounts list
```

Verify that the project is already linked to the intended billing account. If
it is not, copy that account's ID from the preceding command and link it
explicitly:

```powershell
$BillingAccount = '000000-000000-000000'
gcloud billing projects link $ProjectId --billing-account=$BillingAccount
gcloud config set run/region $Region
```

### Google Cloud Console

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and open
   the project selector.
2. Select the existing `uni-rag-agent` project.
3. Open **Billing** from the navigation menu, choose
   **Account management** if necessary, select the intended billing account,
   and click **Link a billing account** only if the project is not already
   linked. Confirm the exact billing account before saving; do not accept an
   accidental default.
4. The Console has no project-wide equivalent to `gcloud config set
   run/region`. Keep `europe-west1` selected whenever a regional resource or
   Cloud Run service is created. The active project selector at the top of the
   Console is the equivalent of `gcloud config set project`.

## 3. Enable APIs and create regional resources

```powershell
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  secretmanager.googleapis.com `
  firestore.googleapis.com

gcloud artifacts repositories create $Repository `
  --repository-format=docker `
  --location=$Region `
  --description='Uni RAG Cloud Run images'

gcloud iam service-accounts create $RuntimeAccount `
  --display-name='Uni RAG runtime'

gcloud firestore databases create `
  --database='(default)' `
  --location=$Region `
  --type=firestore-native `
  --delete-protection
```

Grant the runtime identity only the Firestore entity permissions it needs:

```powershell
$RuntimeEmail = "$RuntimeAccount@$ProjectId.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding $ProjectId `
  --member="serviceAccount:$RuntimeEmail" `
  --role='roles/datastore.user'
```

### Google Cloud Console

1. Open **APIs & Services → Library**, select the new project, search for each
   of the following APIs, and click **Enable**: **Cloud Run Admin API**, **Artifact
   Registry API**, **Cloud Build API**, **Secret Manager API**, and **Cloud
   Firestore API**. Wait for each enablement to finish.
2. Open **Artifact Registry → Repositories → Create repository**. Enter
   `uni-rag`, choose **Docker**, **Standard**, **Region**, and
   `europe-west1`; enter `Uni RAG Cloud Run images` as the description and keep
   the project’s approved encryption and scanning settings. Click **Create**.
3. Open **IAM & Admin → Service Accounts → Create service account**. Enter
   `uni-rag-runtime` as the service-account ID and `Uni RAG runtime` as the
   display name. Finish creation without granting broad project roles. The
   resulting email is
   `uni-rag-runtime@$ProjectId.iam.gserviceaccount.com`.
4. Open **Firestore → Databases → Create a database**. Choose **Firestore
   Native**, database ID **(default)**, and location `europe-west1`, then create
   it. If the creation form exposes **Delete protection**, enable it. If it does
   not, the Console cannot reproduce the CLI's `--delete-protection` flag on
   this screen; use the CLI command already shown or verify the setting on the
   database details page before continuing.
5. Open **IAM & Admin → IAM → Grant access**. Add the runtime service-account
   email as a principal, choose **Cloud Datastore User** (`roles/datastore.user`),
   and save. This is a project-level grant; do not grant Editor or Owner.

## 4. Create secrets without putting values in shell history

In Google Cloud Console, open **Secret Manager → Create secret**. Create these
automatic-replication secrets and add version `1` to each:

- `google-api-key` — Gemini embeddings/planner/answer key.
- `nebius-api-key` — Nebius Token Factory key.
- `demo-token-signing-secret` — a new random value with at least 32 bytes of
  entropy; do not reuse any provider or Turnstile secret.

If you use a second Gemini planner/answer key, also create
`google-api-key-2`. Create `turnstile-secret-key` only in step 7, after the
private hostname is known and the Turnstile widget exists.

Grant access on each created secret (omit the optional second Google key when
unused):

```powershell
$Secrets = @(
  'google-api-key',
  'nebius-api-key',
  'demo-token-signing-secret'
)
foreach ($Secret in $Secrets) {
  gcloud secrets add-iam-policy-binding $Secret `
    --member="serviceAccount:$RuntimeEmail" `
    --role='roles/secretmanager.secretAccessor'
}
```

If you created `google-api-key-2`, grant the same accessor role to it in a
separate command.

### Google Cloud Console: secret creation and access

1. Open **Secret Manager → Secrets → Create secret**. Create each secret one at
   a time with **Automatic** replication, paste its value into **Secret value**,
   and click **Create secret**. Do not paste a secret into a project note,
   browser URL, screenshot, or source file. The required names are
   `google-api-key`, `nebius-api-key`, and `demo-token-signing-secret`; create
   `google-api-key-2` only if it is actually used.
2. For each created secret, select its checkbox, open **Show info panel**, click
   **Add principal**, enter `$RuntimeEmail`, select **Secret Manager Secret
   Accessor**, and save. Repeat for every secret that the Cloud Run revision
   references.
3. For a new version later, open the secret, click **Add new version**, paste
   the value, and save. When configuring Cloud Run, select the numeric version
   `1` (or the new numeric version), not **latest**.

Use numeric versions in Cloud Run (`:1` initially), not `:latest`. Rotation is
an explicit new version followed by a revision update.

## 5. Build the image

```powershell
$ImageTag = Get-Date -Format 'yyyyMMdd-HHmmss'
$Image = "${Region}-docker.pkg.dev/$ProjectId/$Repository/$Service`:$ImageTag"
$BuildAccount = gcloud builds get-default-service-account
gcloud artifacts repositories add-iam-policy-binding $Repository `
  --location=$Region `
  --member="serviceAccount:$BuildAccount" `
  --role='roles/artifactregistry.writer'
gcloud builds submit --tag $Image .
```

### Google Cloud Console

The Console cannot directly submit the current local PowerShell directory in
the same way as `gcloud builds submit .`. In particular, `data/` and
`deployment/assets/` are intentionally ignored by Git, while the selected
SQLite file, vector directory, and staged model must be present in the build
context. For a website-only build, first make the complete repository
**including those three required input paths** available to a source repository
or Cloud Storage source; do not use a normal checkout that omits ignored
deployment inputs.

When the complete build context is in a connected GitHub, Bitbucket, or other
supported source repository:

1. Open **Cloud Build → Triggers → Create trigger**, connect the repository if
   it is not already connected, and choose a manual or tag/branch trigger.
2. Under **Configuration**, choose **Dockerfile**, set the Dockerfile directory
   to the repository root, and set the image destination to the exact `$Image`
   URI (`europe-west1-docker.pkg.dev/$ProjectId/uni-rag/uni-rag-agent:$ImageTag`).
   Set the trigger's region as required by the Console and keep the image tag
   immutable by using a fresh timestamp tag for each build.
3. Select the default Cloud Build service account, or identify the account shown
   by the project and grant it **Artifact Registry Writer** on
   **Artifact Registry → uni-rag → Permissions**. This is the Console
   equivalent of the repository-level `gcloud artifacts ... add-iam-policy-binding`
   command; do not grant the build account project Owner.
4. Save the trigger, open its action menu, choose **Run**, select the source
   revision, and start the build. Monitor **Cloud Build → History** until it is
   successful, then open **Artifact Registry → uni-rag** and verify the image
   and exact tag.

If the staged assets must remain local and uncommitted, keep the CLI build
path above (or run that same command in Cloud Shell after transferring the
build context). A Console trigger backed by a normal repository checkout will
fail at the Docker `COPY` instructions when the ignored model or generated-data
inputs are absent.

## 6. Private first deployment and hostname discovery

The following values preserve the planner/answer models verified by the local
`config check` for this implementation. Keep the two roles separate; if you
change either model later, validate it locally before deploying:

```powershell
$PlannerModel = 'gemini-3-flash-preview'
$AnswerModel = 'gemini-3.5-flash'
$CommonEnv = "UNI_RAG_PUBLIC_DEMO_ENABLED=false,UNI_RAG_LLM_PROVIDER=gemini,UNI_RAG_LLM_MODEL=$PlannerModel,UNI_RAG_ANSWER_LLM_PROVIDER=gemini,UNI_RAG_ANSWER_LLM_MODEL=$AnswerModel,UNI_RAG_EMBEDDING_MODEL=google/gemini-embedding-001,UNI_RAG_FIRESTORE_PROJECT_ID=$ProjectId,UNI_RAG_FIRESTORE_DATABASE=(default),UNI_RAG_PUBLIC_ASK_CAPACITY=2,UNI_RAG_PUBLIC_CAPACITY_WAIT_SECONDS=10,UNI_RAG_PUBLIC_MINUTE_LIMIT=3,UNI_RAG_PUBLIC_CLIENT_DAILY_LIMIT=10,UNI_RAG_PUBLIC_GLOBAL_DAILY_LIMIT=100,UNI_RAG_DEMO_TOKEN_TTL_SECONDS=1800,UNI_RAG_ASK_TIMEOUT_SECONDS=120"
$SecretBindings = 'GOOGLE_API_KEY=google-api-key:1,NEBIUS_API_KEY=nebius-api-key:1,UNI_RAG_DEMO_TOKEN_SIGNING_SECRET=demo-token-signing-secret:1'

# Only when the optional second Google key exists:
# $SecretBindings += ',GOOGLE_API_KEY_2=google-api-key-2:1'

gcloud run deploy $Service `
  --image=$Image `
  --region=$Region `
  --service-account=$RuntimeEmail `
  --no-allow-unauthenticated `
  --execution-environment=gen2 `
  --cpu=2 `
  --memory=8Gi `
  --concurrency=4 `
  --min=0 `
  --max=1 `
  --timeout=300s `
  --cpu-throttling `
  --cpu-boost `
  --port=8080 `
  --startup-probe='timeoutSeconds=10,periodSeconds=10,failureThreshold=24,httpGet.port=8080,httpGet.path=/ready' `
  --readiness-probe='timeoutSeconds=5,periodSeconds=5,failureThreshold=3,httpGet.port=8080,httpGet.path=/ready' `
  --set-env-vars=$CommonEnv `
  --set-secrets=$SecretBindings

$ServiceUrl = gcloud run services describe $Service `
  --region=$Region `
  --format='value(status.url)'
$IdentityToken = gcloud auth print-identity-token
Invoke-RestMethod "$ServiceUrl/ready" -Headers @{ Authorization = "Bearer $IdentityToken" }
```

### Google Cloud Console

1. Open **Cloud Run → Services → Deploy container → Existing container image**.
   Select `europe-west1`, enter `uni-rag-agent` as the service name, and enter
   the exact image URI in `$Image`. Under **Authentication**, select **Require
   authentication**. Do not enable public access yet.
2. Open **Containers, Networking, Security** and configure the container:
   - **Container port:** `8080`.
   - **CPU:** `2`; **Memory:** `8 GiB`.
   - **Request timeout:** `300` seconds; **Maximum concurrent requests:** `4`.
   - **Execution environment:** second generation.
   - **Scaling:** minimum instances `0`, maximum instances `1`.
   - **CPU allocation:** only during request processing (`--cpu-throttling`).
   - Enable **Startup CPU boost** (`--cpu-boost`).
   - **Service account:** select `uni-rag-runtime@$ProjectId.iam.gserviceaccount.com`.
3. In **Variables & Secrets**, add the environment variables from `$CommonEnv`
   as individual rows. The exact rows are:

   | Name | Value |
   | --- | --- |
   | `UNI_RAG_PUBLIC_DEMO_ENABLED` | `false` |
   | `UNI_RAG_LLM_PROVIDER` | `gemini` |
   | `UNI_RAG_LLM_MODEL` | `$PlannerModel` |
   | `UNI_RAG_ANSWER_LLM_PROVIDER` | `gemini` |
   | `UNI_RAG_ANSWER_LLM_MODEL` | `$AnswerModel` |
   | `UNI_RAG_EMBEDDING_MODEL` | `google/gemini-embedding-001` |
   | `UNI_RAG_FIRESTORE_PROJECT_ID` | `$ProjectId` |
   | `UNI_RAG_FIRESTORE_DATABASE` | `(default)` |
   | `UNI_RAG_PUBLIC_ASK_CAPACITY` | `2` |
   | `UNI_RAG_PUBLIC_CAPACITY_WAIT_SECONDS` | `10` |
   | `UNI_RAG_PUBLIC_MINUTE_LIMIT` | `3` |
   | `UNI_RAG_PUBLIC_CLIENT_DAILY_LIMIT` | `10` |
   | `UNI_RAG_PUBLIC_GLOBAL_DAILY_LIMIT` | `100` |
   | `UNI_RAG_DEMO_TOKEN_TTL_SECONDS` | `1800` |
   | `UNI_RAG_ASK_TIMEOUT_SECONDS` | `120` |

4. Still in **Variables & Secrets**, add secret references as environment
   variables. For each row choose **Reference a secret**, select the secret,
   and select version `1`:

   | Environment variable | Secret |
   | --- | --- |
   | `GOOGLE_API_KEY` | `google-api-key:1` |
   | `NEBIUS_API_KEY` | `nebius-api-key:1` |
   | `UNI_RAG_DEMO_TOKEN_SIGNING_SECRET` | `demo-token-signing-secret:1` |

   Add `GOOGLE_API_KEY_2` → `google-api-key-2:1` only when that optional secret
   exists. Do not paste secret values into the environment-variable value
   fields.
5. In **Health checks**, add the startup HTTP probe: path `/ready`, port `8080`,
   timeout `10` seconds, period `10` seconds, and failure threshold `24`. If
   the Console exposes the readiness-probe (Preview) editor, add a readiness
   HTTP probe with path `/ready`, port `8080`, timeout `5` seconds, period `5`
   seconds, and failure threshold `3`. If that Preview control is not shown in
   your Console, leave the CLI command as the authoritative way to configure
   the readiness probe.
6. Click **Create**. Open the new service, wait for the revision to become
   healthy, and copy the service URL from the service details page. This is the
   `$ServiceUrl` used in step 7. The Console shows revision health and logs, but
   a private `/ready` request still needs an authenticated client; the CLI
   identity-token probe above is the reliable private smoke test.

Wait for `/ready` to return `ready`; this proves the copied SQLite/schema,
Chroma client, and eagerly loaded offline EmbeddingGemma are ready without
calling Gemini or Nebius.

## 7. Register the exact Cloud Run hostname with Turnstile

In Cloudflare Dashboard:

1. Open **Turnstile → Add widget**.
2. Name it `Uni RAG Agent Cloud Run` and choose **Managed** mode.
3. Add only the hostname from `$ServiceUrl` (for example,
   `uni-rag-agent-abc123-ew.a.run.app`). Do not include `https://`, a port, or
   a path.
4. Create the widget. Copy the site key. In Secret Manager, create
   `turnstile-secret-key` with the widget secret as version `1`, then grant the
   runtime service account access:

```powershell
gcloud secrets add-iam-policy-binding turnstile-secret-key `
  --member="serviceAccount:$RuntimeEmail" `
  --role='roles/secretmanager.secretAccessor'
```

### Google Cloud Console: finish the secret grant

Open **Secret Manager**, select `turnstile-secret-key`, open **Show info
panel**, click **Add principal**, add `$RuntimeEmail`, choose **Secret Manager
Secret Accessor**, and save. Confirm that the secret has an enabled version `1`
before using it in step 8.

Turnstile tokens are single-use; the app verifies one, then exchanges it for a
signed 30-minute per-tab demo token.

## 8. Enable public mode, then allow unauthenticated Cloud Run access

```powershell
$TurnstileSiteKey = 'YOUR_PUBLIC_TURNSTILE_SITE_KEY'
$PublicEnv = $CommonEnv.Replace(
  'UNI_RAG_PUBLIC_DEMO_ENABLED=false',
  'UNI_RAG_PUBLIC_DEMO_ENABLED=true'
) + ",TURNSTILE_SITE_KEY=$TurnstileSiteKey"
$PublicSecrets = $SecretBindings + ',TURNSTILE_SECRET_KEY=turnstile-secret-key:1'

gcloud run deploy $Service `
  --image=$Image `
  --region=$Region `
  --service-account=$RuntimeEmail `
  --allow-unauthenticated `
  --execution-environment=gen2 `
  --cpu=2 `
  --memory=8Gi `
  --concurrency=4 `
  --min=0 `
  --max=1 `
  --timeout=300s `
  --cpu-throttling `
  --cpu-boost `
  --port=8080 `
  --startup-probe='timeoutSeconds=10,periodSeconds=10,failureThreshold=24,httpGet.port=8080,httpGet.path=/ready' `
  --readiness-probe='timeoutSeconds=5,periodSeconds=5,failureThreshold=3,httpGet.port=8080,httpGet.path=/ready' `
  --set-env-vars=$PublicEnv `
  --set-secrets=$PublicSecrets
```

### Google Cloud Console

1. Open **Cloud Run → Services → uni-rag-agent → Edit and deploy new
   revision**. Keep the same image URI or, preferably, select the exact image
   digest that was verified in the private deployment. Do not accidentally
   deploy `latest` or a different tag.
2. Keep the step 6 container, scaling, service-account, port, timeout,
   concurrency, CPU, memory, and health-check settings. In **Variables &
   Secrets**, change `UNI_RAG_PUBLIC_DEMO_ENABLED` to `true`, keep every other
   `$CommonEnv` value unchanged, and add:

   | Name | Value |
   | --- | --- |
   | `TURNSTILE_SITE_KEY` | `$TurnstileSiteKey` |

3. Keep the three step 6 secret references and add
   `TURNSTILE_SECRET_KEY` → `turnstile-secret-key` version `1`. Keep
   `GOOGLE_API_KEY_2` only if it is used.
4. Under **Authentication**, select **Allow unauthenticated invocations**.
   This grants the special principal `allUsers` the Cloud Run Invoker role,
   which is the Console equivalent of `--allow-unauthenticated`. If an
   organization policy prevents this selection, stop and resolve that policy;
   do not compensate by granting a broad user or service-account role.
5. Click **Deploy**. Wait for the new revision to become healthy and confirm
   that it receives 100% of traffic. Open the service URL only after this
   public revision is serving.

Concurrency `4` is intentionally higher than the app ask capacity `2`; the two
remaining request slots keep progress, cancellation, CAPTCHA exchange, and
static assets responsive. Cloud Run's `300s` timeout is deliberate platform
headroom; the app's `120s` timeout is the operative ask limit.

## 9. Create a $10 monthly budget alert and provider limits

Find the project resource name and create alerts at 50%, 80%, and 100%:

```powershell
$ProjectNumber = gcloud projects describe $ProjectId --format='value(projectNumber)'
gcloud billing budgets create `
  --billing-account=$BillingAccount `
  --display-name='Uni RAG Agent monthly alert' `
  --budget-amount=10USD `
  --calendar-period=month `
  --filter-projects="projects/$ProjectNumber" `
  --threshold-rule=percent=0.5 `
  --threshold-rule=percent=0.8 `
  --threshold-rule=percent=1.0
```

### Google Cloud Console

1. Select `$ProjectId` in the project selector before opening **Billing**.
   Go to **Billing → Budgets & alerts → Create budget**. If the Console opens
   at billing-account scope, select the billing account linked to this project
   and then select the project in the budget scope.
2. Name the budget `Uni RAG Agent monthly alert`, set the amount to `10 USD`,
   choose a **Monthly** calendar period, and scope it to `$ProjectId` only.
3. In the threshold section, add alert thresholds at **50%**, **80%**, and
   **100%**. Keep the notification recipients appropriate for the billing
   account and click **Finish** or **Save**.
4. Confirm the budget's project filter is this project, not the entire billing
   account. A budget sends notifications; it is not a hard spending cap. Set
   low Gemini and Nebius provider-side quotas in their respective provider
   consoles as well.

Budget alerts notify; they are not hard spending caps. Also set low Gemini and
Nebius provider-side quotas where their consoles allow it.

## 10. Launch verification

Open `$ServiceUrl`, complete Turnstile, and run an ask with each embedding
profile. `/ready` must return `ready`; the rest of the functional and resource
verification is operator-owned.

### Google Cloud Console

1. Open **Cloud Run → Services → uni-rag-agent**, confirm the serving revision,
   region, URL, authentication setting, and traffic split.
2. Open the service URL from the details page in a browser. Complete Turnstile,
   run an ask with each supported embedding profile, and verify the response in
   the UI.
3. Use the service's **Logs** tab for startup or request failures and the
   **Metrics** tab for instance count, latency, and request errors. Confirm that
   `/ready` returns `ready` and that no unexpected revision receives traffic.

## 11. Rotation, rollback, reset, and clean redeployment

- **Rotate a provider/Turnstile/signing secret:** add a new numeric secret
  version, grant no new role, deploy a revision referencing that version, test,
  then disable the old version. Rotating the signing secret immediately
  invalidates existing demo tokens.

  **Console:** In **Secret Manager**, open the secret and click **Add new
  version**. Then open **Cloud Run → uni-rag-agent → Edit and deploy new
  revision**, change only the referenced secret version to the new number, and
  deploy. Test the service before opening the old secret version's action menu
  and choosing **Disable**. Do not delete the old version until rollback is no
  longer needed.
- **Reset quotas:** quota documents live in `demo_quota_global`,
  `demo_quota_clients`, `demo_quota_minutes`, and
  `demo_quota_reservations`. Prefer natural UTC-day/minute expiry. Manual
  deletion is an operator action and should be limited to a confirmed test
  client/day.

  **Console:** Open **Firestore → Data**, select the `(default)` database, and
  inspect the named collections before deleting anything. Delete only confirmed
  test documents for the intended client/day; do not delete an entire
  collection or production quota set from the Console.
- **Rollback:** list revisions with `gcloud run revisions list
  --service=$Service --region=$Region`; send traffic to a known-good revision
  with `gcloud run services update-traffic $Service --region=$Region
  --to-revisions=REVISION=100`.

  **Console:** Open **Cloud Run → Services → uni-rag-agent → Revisions**. Use
  **Manage traffic**, assign 100% to the known-good revision, set 0% for the
  bad revision, and save. Verify the active revision and service URL, then
  inspect logs and `/ready`.
- **Clean redeploy:** rebuild `data/` from `Courses/`, restage
  `deployment/assets/models/embeddinggemma-300m`, build a new immutable image
  tag, deploy privately, verify `/ready`, then deploy the same digest publicly.

  **Console:** After the local rebuild and asset staging, make the complete
  build context available to the Cloud Build source as described in step 5.
  Run the build, verify the new image digest in **Artifact Registry**, use
  **Cloud Run → Edit and deploy new revision** with authentication required for
  the private deployment, verify `/ready`, then deploy the same digest again
  with public access and the Turnstile settings from step 8.
