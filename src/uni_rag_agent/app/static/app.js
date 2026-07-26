const shell = document.querySelector("#shell");
const hero = document.querySelector("#hero");
const form = document.querySelector("#ask-form");
const queryInput = document.querySelector("#query");
const queryCount = document.querySelector("#query-count");
const askButton = document.querySelector("#ask-button");
const statusBox = document.querySelector("#status");
const result = document.querySelector("#result");
const detailsToggle = document.querySelector("#details-toggle");
const detailsSection = document.querySelector("#details");
const historySection = document.querySelector("#history");
const historyList = document.querySelector("#history-list");
const clearHistoryButton = document.querySelector("#clear-history");
const activeSessionLabel = document.querySelector("#active-session-label");
const sessionTag = activeSessionLabel.closest(".session-tag");
const newSessionButton = document.querySelector("#new-session");
const cancelRequestButton = document.querySelector("#cancel-request");
const settingsButton = document.querySelector("#settings-button");
const settingsDialog = document.querySelector("#settings-dialog");
const settingsForm = document.querySelector("#settings-form");
const settingsFields = document.querySelector("#settings-fields");
const settingsStatus = document.querySelector("#settings-status");
const settingsChanged = document.querySelector("#settings-changed");
const settingsSaveButton = document.querySelector("#settings-save");
const settingsResetButton = document.querySelector("#settings-reset");
const settingsCloseButton = document.querySelector("#settings-close");
const settingsNote = document.querySelector("#settings-note");
const themeToggle = document.querySelector("#theme-toggle");
const themeIcon = document.querySelector("#theme-icon");
const themeColor = document.querySelector("#theme-color");
const turnstilePanel = document.querySelector("#turnstile-panel");
const turnstileWidget = document.querySelector("#turnstile-widget");
const indexChip = document.querySelector("#index-chip");
const embeddingLoadingIndicator = document.querySelector("#embedding-loading-indicator");
const embeddingLoadingLabel = document.querySelector("#embedding-loading-label");
const progressPanel = document.querySelector("#progress-panel");
const progressTitle = document.querySelector("#progress-title");
const progressStage = document.querySelector("#progress-stage");
const stageList = document.querySelector("#stage-list");
const questionIndexLabel = document.querySelector("#question-index");
const questionMetaDetail = document.querySelector("#question-meta-detail");
const askedQuery = document.querySelector("#asked-query");
const receiptBox = document.querySelector("#receipt");
const answerNotice = document.querySelector("#answer-notice");
const answerCard = document.querySelector("#answer-card");
const answerState = document.querySelector("#answer-state");
const answerRoot = document.querySelector("#answer-text");
const evidenceCited = document.querySelector("#evidence-cited");
const searchedPanel = document.querySelector("#searched-panel");
const browserState = window.UniRagBrowserState;

const SESSIONS_KEY = "uni-rag-sessions";
const ACTIVE_KEY = "uni-rag-active-session";
const DETAILS_KEY = "uni-rag-details";
const PUBLIC_SETTINGS_KEY = "uni-rag-public-settings";
const DEMO_TOKEN_KEY = "uni-rag-demo-token";
const DEMO_TOKEN_EXP_KEY = "uni-rag-demo-token-exp";
const THEME_KEY = "uni-rag-theme";
let current = null;
let currentPacket = null;
let currentMeta = null;
let packetLoadedFor = null;
let appMode = "local";
let sessionStore = localStorage;
let themeStore = localStorage;
let sessions = [];
let activeSessionId = null;
let activeSessionLive = false;
let activeRequest = null;
let submissionPending = false;
let settingsPayload = null;
let turnstilePromise = null;
let quotaRemaining = null;
let settingsOperation = null;
let settingsLoadFailed = false;

initializeTheme();
resizeQueryInput();
updateQueryCount();
initializeApp();

window.addEventListener("error", (event) => recoverFromClientError(event.error || event.message));
window.addEventListener("unhandledrejection", (event) => recoverFromClientError(event.reason));

function recoverFromClientError(error) {
  const message = error instanceof Error ? error.message : "An unexpected browser error occurred.";
  if (activeRequest) {
    stopRequestFeedback(activeRequest);
    activeRequest = null;
  }
  submissionPending = false;
  setBusy(false);
  setStatus(`The page recovered from an error: ${message}`, "error");
  renderHistory();
}

function initializeTheme() {
  const savedTheme = themeStore.getItem(THEME_KEY);
  applyTheme(savedTheme === "dark" ? "dark" : "light");
}

function applyTheme(theme) {
  const nextTheme = theme === "dark" ? "dark" : "light";
  const switchTo = nextTheme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = nextTheme;
  themeIcon.textContent = nextTheme === "dark" ? "☀" : "☾";
  themeToggle.setAttribute("aria-label", `Switch to ${switchTo} theme`);
  themeToggle.title = `Switch to ${switchTo} theme`;
  themeColor.setAttribute("content", nextTheme === "dark" ? "#1c1916" : "#fdfcf9");
}

function resizeQueryInput() {
  queryInput.style.height = "auto";
  queryInput.style.height = `${queryInput.scrollHeight}px`;
}

function updateQueryCount() {
  const limit = Number(queryInput.maxLength) || 10_000;
  const length = queryInput.value.length;
  queryCount.textContent = length >= limit * 0.8 ? `${length.toLocaleString()} / ${limit.toLocaleString()} characters` : "";
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

async function initializeApp() {
  setBusy(true, "Loading application mode…");
  try {
    settingsPayload = await requestJson("/api/settings");
    appMode = settingsPayload.mode === "public" ? "public" : "local";
    sessionStore = browserState.selectStore(appMode, localStorage, sessionStorage);
    themeStore = browserState.selectStore(appMode, localStorage, sessionStorage);
    initializeTheme();
    sessions = loadSessions();
    activeSessionId = sessionStore.getItem(ACTIVE_KEY);
    if (activeSessionId && !findSession(activeSessionId)) activeSessionId = null;
    activeSessionLive = activeSessionId ? null : false;
    detailsToggle.checked = sessionStore.getItem(DETAILS_KEY) === "1";
    if (appMode === "public") {
      queryInput.maxLength = settingsPayload.public_limits?.query_max_chars || 4000;
      updateQueryCount();
      settingsNote.textContent = "Settings are private to this tab and are sent only with your next question. They are not written to the server.";
      hydratePublicSettingsPayload();
    }
    applyDetailsVisibility();
    renderSessionState();
    renderHistory();
    await restoreActiveSession();
    clearStatus();
  } catch (error) {
    setStatus(`Could not initialize the application: ${error.message}`, "error");
  } finally {
    setBusy(false);
  }
}

detailsToggle.addEventListener("change", () => {
  sessionStore.setItem(DETAILS_KEY, detailsToggle.checked ? "1" : "0");
  applyDetailsVisibility();
});

themeToggle.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  themeStore.setItem(THEME_KEY, nextTheme);
  applyTheme(nextTheme);
});

newSessionButton.addEventListener("click", () => {
  if (activeRequest || submissionPending) return;
  activeSessionId = null;
  activeSessionLive = false;
  current = null;
  packetLoadedFor = null;
  sessionStore.removeItem(ACTIVE_KEY);
  clearResult();
  clearStatus();
  queryInput.value = "";
  resizeQueryInput();
  updateQueryCount();
  renderSessionState();
  renderHistory();
  queryInput.focus();
});

clearHistoryButton.addEventListener("click", () => {
  if (activeRequest || submissionPending) return;
  sessions = browserState.clearSessions(sessionStore, SESSIONS_KEY, ACTIVE_KEY);
  activeSessionId = null;
  activeSessionLive = false;
  queryInput.value = "";
  resizeQueryInput();
  updateQueryCount();
  clearResult();
  clearStatus();
  renderSessionState();
  renderHistory();
  queryInput.focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "auto" });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (activeRequest || submissionPending) return;
  if (activeSessionLive === null) {
    setStatus("Wait for the active-session check to finish before asking.", "working");
    return;
  }
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Enter a question before asking.", "error");
    queryInput.focus();
    return;
  }
  submissionPending = true;
  setBusy(true);
  queryInput.value = "";
  resizeQueryInput();
  updateQueryCount();
  if (appMode === "public") {
    try {
      await ensureDemoToken();
    } catch (error) {
      setStatus(error.message, "error");
      submissionPending = false;
      setBusy(false);
      return;
    }
  }
  const requestId = generateRequestId();
  const controller = new AbortController();
  const request = {
    requestId,
    controller,
    cancelled: false,
    progressTimer: null,
    elapsedTimer: null,
    startedAt: Date.now(),
    stages: defaultStages(),
    activeIndex: -1,
    activeStartedAt: null,
    currentPhase: null,
  };
  activeRequest = request;
  setBusy(true);
  const sessionId = activeSessionId && activeSessionLive ? activeSessionId : generateSessionId();
  try {
    beginQuestion(query);
    startRequestFeedback(request);
    current = await requestJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        request_id: requestId,
        ...(appMode === "public" ? { retrieval_settings: publicRequestSettings() } : {}),
      }),
      signal: controller.signal,
    });
    completeStages(request);
    packetLoadedFor = null;
    activeSessionId = sessionId;
    activeSessionLive = true;
    sessionStore.setItem(ACTIVE_KEY, sessionId);
    recordTurn(sessionId, query, current);
    if (appMode === "public" && current.remaining) quotaRemaining = current.remaining;
    renderSessionState();
    renderHistory();
    renderAnswer(current, query, {
      stages: request.stages,
      elapsedSeconds: (Date.now() - request.startedAt) / 1000,
      at: Date.now(),
      turnIndex: findSession(sessionId)?.turns.length || 1,
    });
    queryInput.value = "";
    resizeQueryInput();
    updateQueryCount();
    clearStatus();
  } catch (error) {
    if (!request.cancelled) {
      setStatus(error.message, "error");
      showRequestFailure(query, error);
    } else {
      clearResult();
    }
  } finally {
    if (activeRequest === request) {
      stopRequestFeedback(request);
      setBusy(false);
    }
    submissionPending = false;
  }
});

cancelRequestButton.addEventListener("click", async () => {
  const request = activeRequest;
  if (!request) return;
  cancelRequestButton.disabled = true;
  try {
    const outcome = await requestJson(`/api/asks/${request.requestId}/cancel`, {
      method: "POST",
    });
    if (outcome.cancelled) {
      request.cancelled = true;
      setStatus("Request cancelled. Any in-flight work will finish without saving an answer.", "cancelled");
      request.controller.abort();
    } else {
      setStatus("The request completed before it could be cancelled.", "working");
    }
  } catch (error) {
    setStatus(`Could not cancel the request: ${error.message}`, "error");
  } finally {
    if (activeRequest === request && !request.cancelled) cancelRequestButton.disabled = false;
  }
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!askButton.disabled) form.requestSubmit(askButton);
  }
});

queryInput.addEventListener("input", () => {
  resizeQueryInput();
  updateQueryCount();
});

/* ---------- retrieval settings dialog ---------- */

const SETTING_LABELS = {
  embedding_model: "Embedding model",
  keyword_top_k: "Keyword top-k",
  semantic_top_k: "Semantic top-k",
  metadata_top_k: "Metadata top-k",
  final_top_k: "Final evidence items",
  rrf_k: "RRF rank constant",
  semantic_query_limit: "Semantic queries per ask",
  filename_fuzzy_threshold: "Filename fuzzy threshold",
  path_fuzzy_threshold: "Path fuzzy threshold",
  evidence_max_tokens: "Evidence token budget",
  query_plan_min_confidence: "Minimum plan confidence",
};
const FLOAT_SETTINGS = new Set(["query_plan_min_confidence"]);
// Grouped by what each value actually affects, as rows of same-width fields.
const SETTING_GROUPS = [
  {
    label: "Retrieval",
    rows: [["embedding_model"], ["keyword_top_k", "semantic_top_k", "metadata_top_k"]],
  },
  { label: "Fusion", rows: [["rrf_k", "semantic_query_limit", "final_top_k"]] },
  {
    label: "Filename matching",
    rows: [["filename_fuzzy_threshold", "path_fuzzy_threshold"]],
  },
  {
    label: "Generation",
    rows: [["evidence_max_tokens", "query_plan_min_confidence"]],
  },
];

settingsButton.addEventListener("click", async () => {
  settingsDialog.showModal();
  settingsCloseButton.focus();
  if (settingsOperation) {
    setSettingsControlsDisabled(true);
    return;
  }
  clearSettingsStatus();
  clearSettingsErrors();
  settingsLoadFailed = false;
  settingsSaveButton.disabled = true;
  settingsResetButton.disabled = true;
  settingsFields.replaceChildren(emptyMessage("Loading current settings…"));
  try {
    if (!settingsPayload || appMode === "local") {
      settingsPayload = await requestJson("/api/settings");
    }
    if (appMode === "public") hydratePublicSettingsPayload();
    renderSettingsForm(settingsPayload);
  } catch (error) {
    settingsLoadFailed = true;
    settingsFields.replaceChildren(
      emptyMessage(`Could not load settings: ${error.message}`),
    );
  } finally {
    setSettingsControlsDisabled(settingsLoadFailed);
  }
});

settingsCloseButton.addEventListener("click", requestSettingsDialogClose);
settingsDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  requestSettingsDialogClose();
});
settingsDialog.addEventListener("close", () => settingsButton.focus());

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!settingsPayload || settingsOperation || settingsLoadFailed) return;
  const changes = collectSettingsChanges();
  if (changes === null) return;
  if (!Object.keys(changes).length) {
    settingsDialog.close();
    return;
  }
  await submitSettings(changes, "Settings saved. They apply from your next question.");
});

settingsResetButton.addEventListener("click", async () => {
  if (!settingsPayload || settingsOperation || settingsLoadFailed) return;
  if (!window.confirm("Reset all retrieval settings to their server defaults?")) return;
  const changes = {};
  Object.keys(SETTING_LABELS).forEach((name) => {
    if (settingsPayload.overrides[name] !== undefined) changes[name] = null;
  });
  if (!Object.keys(changes).length) {
    settingsDialog.close();
    return;
  }
  await submitSettings(changes, "All settings now follow the server configuration.");
});

async function submitSettings(changes, successMessage) {
  if (settingsOperation) return;
  const operation = { changes, preparingModel: null };
  settingsOperation = operation;
  setSettingsControlsDisabled(true);
  try {
    const localModel = browserState.localEmbeddingPreparationModel(
      settingsPayload,
      changes,
    );
    if (localModel) {
      operation.preparingModel = localModel;
      if (appMode === "public") await ensureDemoToken();
      setEmbeddingLoadingIndicator(localModel);
      setSettingsStatus(`Preparing ${localModel}…`, "working");
      await requestJson("/api/embedding-profiles/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ embedding_model: localModel }),
      });
    }
    if (appMode === "public") {
      const stored = Object.fromEntries(
        Object.entries(changes).filter(([, value]) => value !== null),
      );
      sessionStorage.setItem(PUBLIC_SETTINGS_KEY, JSON.stringify(stored));
      settingsPayload = { ...settingsPayload, overrides: stored };
    } else {
      settingsPayload = await requestJson("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });
    }
    renderSettingsForm(settingsPayload);
    setSettingsStatus(successMessage, "ok");
    settingsDialog.close();
  } catch (error) {
    const message = operation.preparingModel
      ? `Could not prepare ${operation.preparingModel}. Choose another model or try again.`
      : error.message;
    setSettingsStatus(message, operation.preparingModel ? "prepare-error" : "error");
  } finally {
    if (settingsOperation === operation) {
      setEmbeddingLoadingIndicator(null);
      settingsOperation = null;
      setSettingsControlsDisabled(false);
    }
  }
}

function setSettingsControlsDisabled(disabled) {
  settingsSaveButton.disabled = disabled;
  settingsResetButton.disabled = disabled;
}

function requestSettingsDialogClose() {
  if (!settingsOperation && hasUnsavedSettingsChanges()) {
    if (!window.confirm("Discard unsaved settings changes?")) return;
  }
  settingsDialog.close();
}

function collectSettingsChanges() {
  clearSettingsErrors();
  const changes = {};
  let firstInvalid = null;
  Object.keys(SETTING_LABELS).forEach((name) => {
    const input = settingsForm.querySelector(`[name="${name}"]`);
    if (!input) return;
    const raw = input.value.trim();
    const previous = settingsPayload.overrides[name] ?? null;
    let value = raw === "" ? null : raw;
    if (name !== "embedding_model" && raw !== "") {
      const integer = !FLOAT_SETTINGS.has(name);
      const validFormat = integer
        ? /^-?\d+$/.test(raw)
        : /^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(raw);
      const numeric = Number(raw);
      const limits = settingsPayload.limits?.[name];
      if (!validFormat || !Number.isFinite(numeric)) {
        const message = `${SETTING_LABELS[name]} must be ${integer ? "an integer" : "a number"}.`;
        setSettingsFieldError(name, message);
        firstInvalid = firstInvalid || input;
        return;
      }
      if (limits && (numeric < limits.min || numeric > limits.max)) {
        const message = `${SETTING_LABELS[name]} must be between ${formatNumber(limits.min)} and ${formatNumber(limits.max)}.`;
        setSettingsFieldError(name, message);
        firstInvalid = firstInvalid || input;
        return;
      }
      value = numeric;
    }
    if (value !== previous) changes[name] = value;
  });
  if (firstInvalid) {
    setSettingsStatus("Correct the highlighted setting before saving.", "error");
    firstInvalid.focus();
    return null;
  }
  return changes;
}

function setEmbeddingLoadingIndicator(modelName) {
  const label = modelName === "google/embeddinggemma-300m"
    ? "Loading Gemma…"
    : "Loading embedding model…";
  embeddingLoadingLabel.textContent = label;
  embeddingLoadingIndicator.setAttribute("aria-label", label);
  embeddingLoadingIndicator.hidden = !modelName;
}

function renderSettingsForm(payload) {
  const fragment = document.createDocumentFragment();
  SETTING_GROUPS.forEach((group) => {
    const section = document.createElement("section");
    section.className = "settings-group";
    const label = document.createElement("div");
    label.className = "settings-group-label";
    label.textContent = group.label;
    section.append(label);
    group.rows.forEach((row) => {
      const grid = document.createElement("div");
      grid.className = `settings-grid cols-${row.length}`;
      row.forEach((name) => {
        grid.append(
          name === "embedding_model"
            ? buildEmbeddingModelField(payload)
            : buildNumericField(payload, name),
        );
      });
      section.append(grid);
    });
    fragment.append(section);
  });
  settingsFields.replaceChildren(fragment);
  updateChangedMarkers();
}

function buildEmbeddingModelField(payload) {
  const field = settingsField("embedding_model");
  const select = document.createElement("select");
  select.name = "embedding_model";
  select.id = "setting-embedding_model";
  const fallback = document.createElement("option");
  fallback.value = "";
  const configured = payload.defaults.embedding_model;
  fallback.textContent = configured
    ? `Server default — ${configured}`
    : "Server default (not set)";
  select.append(fallback);
  (payload.embedding_model_profiles || []).forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.model_name;
    option.textContent = `${profile.model_name} · ${profile.provider} · ${profile.dimension}d`;
    select.append(option);
  });
  select.value = payload.overrides.embedding_model || "";
  select.addEventListener("change", () => {
    clearSettingsFieldError("embedding_model");
    updateChangedMarkers();
  });
  field.append(select);
  field.append(field.querySelector(".settings-field-error"));
  // The blank option already names the server default, so no hint row here.
  return field;
}

function buildNumericField(payload, name) {
  const field = settingsField(name);
  const input = document.createElement("input");
  input.type = "text";
  input.inputMode = FLOAT_SETTINGS.has(name) ? "decimal" : "numeric";
  input.name = name;
  input.id = `setting-${name}`;
  const limits = payload.limits?.[name];
  if (limits) {
    input.min = limits.min;
    input.max = limits.max;
  }
  input.autocomplete = "off";
  input.placeholder = `default ${formatValue(payload.defaults[name])}`;
  const override = payload.overrides[name];
  input.value = override === undefined || override === null ? "" : override;
  input.addEventListener("input", () => {
    clearSettingsFieldError(name);
    updateChangedMarkers();
  });
  field.append(input);
  const hint = `default ${formatValue(payload.defaults[name])}${
    limits ? ` · ${formatNumber(limits.min)}–${formatNumber(limits.max)}` : ""
  }`;
  field.append(hintRow(name, payload, hint));
  input.setAttribute("aria-describedby", `setting-${name}-hint setting-${name}-error`);
  field.append(field.querySelector(".settings-field-error"));
  return field;
}

function hintRow(name, payload, text) {
  const hint = document.createElement("small");
  hint.className = "settings-hint";
  hint.id = `setting-${name}-hint`;
  const value = document.createElement("span");
  value.textContent = text;
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "reset";
  reset.textContent = "reset";
  reset.hidden = true;
  reset.addEventListener("click", () => {
    const input = settingsForm.querySelector(`[name="${name}"]`);
    if (!input) return;
    input.value = "";
    clearSettingsFieldError(name);
    updateChangedMarkers();
  });
  hint.append(value, reset);
  return hint;
}

function settingsField(name) {
  const wrap = document.createElement("div");
  wrap.className = "settings-field";
  wrap.dataset.field = name;
  const label = document.createElement("label");
  label.htmlFor = `setting-${name}`;
  label.textContent = SETTING_LABELS[name];
  const dot = document.createElement("span");
  dot.className = "changed-dot";
  dot.hidden = true;
  label.append(dot);
  wrap.append(label);
  const error = document.createElement("small");
  error.id = `setting-${name}-error`;
  error.className = "settings-field-error";
  error.hidden = true;
  wrap.append(error);
  return wrap;
}

function updateChangedMarkers() {
  if (!settingsPayload) return;
  let changed = 0;
  Object.keys(SETTING_LABELS).forEach((name) => {
    const input = settingsForm.querySelector(`[name="${name}"]`);
    const field = settingsForm.querySelector(`[data-field="${name}"]`);
    if (!input || !field) return;
    const current = settingsPayload.overrides[name] ?? null;
    const isChanged = input.value.trim() !== (current === null ? "" : String(current));
    const hasOverrideValue = input.value.trim() !== "";
    if (isChanged) changed += 1;
    field.classList.toggle("changed", isChanged);
    const dot = field.querySelector(".changed-dot");
    const reset = field.querySelector(".reset");
    if (dot) dot.hidden = !isChanged;
    if (reset) reset.hidden = !hasOverrideValue;
  });
  settingsChanged.hidden = changed === 0;
  settingsChanged.textContent = `${changed} unsaved change${changed === 1 ? "" : "s"}`;
}

function hasUnsavedSettingsChanges() {
  if (!settingsPayload || settingsLoadFailed) return false;
  return Object.keys(SETTING_LABELS).some((name) => {
    const input = settingsForm.querySelector(`[name="${name}"]`);
    const current = settingsPayload.overrides[name] ?? null;
    return input && input.value.trim() !== (current === null ? "" : String(current));
  });
}

function clearSettingsErrors() {
  Object.keys(SETTING_LABELS).forEach(clearSettingsFieldError);
}

function clearSettingsFieldError(name) {
  const input = settingsForm.querySelector(`[name="${name}"]`);
  const field = settingsForm.querySelector(`[data-field="${name}"]`);
  const error = document.querySelector(`#setting-${name}-error`);
  if (input) {
    input.removeAttribute("aria-invalid");
    input.setAttribute("aria-describedby", `setting-${name}-hint setting-${name}-error`);
  }
  if (field) field.classList.remove("has-error");
  if (error) {
    error.hidden = true;
    error.textContent = "";
  }
}

function setSettingsFieldError(name, message) {
  const input = settingsForm.querySelector(`[name="${name}"]`);
  const field = settingsForm.querySelector(`[data-field="${name}"]`);
  const error = document.querySelector(`#setting-${name}-error`);
  if (input) {
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", `setting-${name}-hint setting-${name}-error`);
  }
  if (field) field.classList.add("has-error");
  if (error) {
    error.textContent = message;
    error.hidden = false;
  }
}

function setSettingsStatus(message, kind) {
  settingsStatus.hidden = false;
  settingsStatus.textContent = message;
  settingsStatus.className = `settings-status ${kind || ""}`;
  settingsStatus.setAttribute("aria-live", kind === "error" || kind === "prepare-error" ? "assertive" : "polite");
}

function clearSettingsStatus() {
  settingsStatus.hidden = true;
  settingsStatus.textContent = "";
}

function hydratePublicSettingsPayload() {
  if (!settingsPayload || appMode !== "public") return;
  let stored = {};
  try {
    const parsed = JSON.parse(sessionStorage.getItem(PUBLIC_SETTINGS_KEY) || "{}");
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) stored = parsed;
  } catch {
    stored = {};
  }
  settingsPayload = { ...settingsPayload, overrides: stored };
}

function publicRequestSettings() {
  if (appMode !== "public") return null;
  hydratePublicSettingsPayload();
  return { ...(settingsPayload?.overrides || {}) };
}

async function ensureDemoToken() {
  if (appMode !== "public") return null;
  const state = browserState.tokenState(
    sessionStorage,
    DEMO_TOKEN_KEY,
    DEMO_TOKEN_EXP_KEY,
    Math.floor(Date.now() / 1000),
  );
  if (state.valid) return state.token;
  if (state.renewalRequired && activeSessionId) {
    detachActiveSession();
    renderSessionState();
  }
  sessionStorage.removeItem(DEMO_TOKEN_KEY);
  sessionStorage.removeItem(DEMO_TOKEN_EXP_KEY);
  if (!turnstilePromise) turnstilePromise = runTurnstile();
  try {
    return await turnstilePromise;
  } finally {
    turnstilePromise = null;
  }
}

function runTurnstile() {
  const sitekey = settingsPayload?.turnstile_site_key;
  if (!sitekey) return Promise.reject(new Error("Public verification is not configured."));
  turnstilePanel.hidden = false;
  setStatus("Complete the one-time verification for this tab.", "working");
  return loadTurnstileScript().then(() => new Promise((resolve, reject) => {
    turnstileWidget.replaceChildren();
    window.turnstile.render(turnstileWidget, {
      sitekey,
      callback: async (turnstileToken) => {
        try {
          const session = await requestJson("/api/demo/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ turnstile_token: turnstileToken }),
          });
          sessionStorage.setItem(DEMO_TOKEN_KEY, session.demo_token);
          sessionStorage.setItem(DEMO_TOKEN_EXP_KEY, String(session.expires_at));
          turnstilePanel.hidden = true;
          clearStatus();
          resolve(session.demo_token);
        } catch (error) {
          reject(error);
        }
      },
      "error-callback": () => reject(new Error("Verification could not be completed.")),
      "expired-callback": () => reject(new Error("Verification expired. Please try again.")),
    });
  }));
}

function loadTurnstileScript() {
  if (window.turnstile) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const existing = document.querySelector("script[data-uni-rag-turnstile]");
    if (existing) {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", reject, { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    script.async = true;
    script.defer = true;
    script.dataset.uniRagTurnstile = "1";
    script.addEventListener("load", resolve, { once: true });
    script.addEventListener("error", () => reject(new Error("Verification service could not be loaded.")), { once: true });
    document.head.append(script);
  });
}

function renderTopChip() {
  if (activeRequest) {
    indexChip.hidden = false;
    indexChip.className = "chip live";
    const elapsed = (Date.now() - activeRequest.startedAt) / 1000;
    indexChip.textContent = `Working · ${elapsed.toFixed(1)}s`;
    return;
  }
  if (appMode === "public") {
    if (!quotaRemaining) {
      indexChip.hidden = true;
      return;
    }
    indexChip.hidden = false;
    const low = quotaRemaining.client_day <= 2;
    indexChip.className = low ? "chip warn" : "chip";
    indexChip.textContent = `${quotaRemaining.client_day} asks left today · ${quotaRemaining.minute} this minute`;
    return;
  }
  indexChip.hidden = true;
}

/* ---------- session log (client-side; the server keeps no session listing) ---------- */

function loadSessions() {
  return browserState.loadSessions(sessionStore, SESSIONS_KEY);
}

function saveSessions() {
  const limit = appMode === "public" ? 20 : 50;
  sessions = browserState.saveSessions(
    sessionStore,
    SESSIONS_KEY,
    sessions,
    limit,
    true,
  );
}

function findSession(id) {
  return sessions.find((session) => session.id === id) || null;
}

function generateSessionId() {
  return browserState.randomId(crypto);
}

function generateRequestId() {
  return browserState.randomId(crypto);
}

function recordTurn(sessionId, query, answerPayload) {
  let session = findSession(sessionId);
  if (!session) {
    session = { id: sessionId, title: query, turns: [], updated: 0 };
    sessions.unshift(session);
  }
  session.turns.push({ query, answer_id: answerPayload.answer_id, at: Date.now() });
  if (appMode === "public") {
    session.latest = { query, payload: answerPayload };
    session.settings = publicRequestSettings();
  }
  session.updated = Date.now();
  sessions = [session, ...sessions.filter((entry) => entry.id !== sessionId)];
  saveSessions();
}

function renderSessionState() {
  const session = activeSessionId ? findSession(activeSessionId) : null;
  if (session) {
    const prefix = activeSessionLive === true
      ? "Continuing"
      : activeSessionLive === null
        ? "Checking session"
        : "Session expired";
    activeSessionLabel.textContent = `${prefix}: ${truncate(session.title, 40)}`;
    sessionTag.classList.toggle("live", activeSessionLive === true);
    newSessionButton.hidden = false;
  } else {
    activeSessionLabel.textContent = "New session";
    sessionTag.classList.remove("live");
    newSessionButton.hidden = true;
  }
}

function renderHistory() {
  historyList.replaceChildren();
  const previous = sessions.filter((session) => session.id !== activeSessionId);
  historySection.hidden = !sessions.length;
  previous.forEach((session) => {
    const item = document.createElement("article");
    item.className = "history-item";

    const main = document.createElement("button");
    main.type = "button";
    main.className = "history-main";
    main.title = "Resume this session and reload its latest answer";
    const title = document.createElement("strong");
    title.textContent = truncate(session.title, 90);
    const meta = document.createElement("small");
    const count = session.turns.length;
    meta.textContent = `${count} question${count === 1 ? "" : "s"} · ${relativeTime(session.updated)}`;
    main.append(title, meta);
    main.disabled = Boolean(activeRequest || submissionPending);
    main.addEventListener("click", () => {
      if (!activeRequest && !submissionPending) resumeSession(session);
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "history-remove";
    remove.setAttribute("aria-label", "Remove this session from the log");
    remove.textContent = "✕";
    remove.disabled = Boolean(activeRequest || submissionPending);
    remove.addEventListener("click", () => {
      if (activeRequest || submissionPending) return;
      sessions = sessions.filter((entry) => entry.id !== session.id);
      saveSessions();
      renderHistory();
    });

    item.append(main, remove);
    historyList.append(item);
  });
}

async function resumeSession(session) {
  clearResult();
  activeSessionId = session.id;
  activeSessionLive = null;
  sessionStore.setItem(ACTIVE_KEY, session.id);
  renderSessionState();
  renderHistory();
  const lastTurn = session.turns[session.turns.length - 1];
  const resumeMeta = {
    stages: null,
    elapsedSeconds: null,
    at: session.updated,
    turnIndex: session.turns.length,
  };
  if (appMode === "public") {
    const initialDecision = browserState.publicResumeDecision(session, {
      tokenRenewed: false,
      serverLive: false,
    });
    if (initialDecision.remove) {
      removeSession(session.id);
      setStatus("This tab history entry is incomplete, so it was removed.", "error");
      renderSessionState();
      renderHistory();
      return;
    }
    const latest = session.latest;
    current = initialDecision.payload;
    packetLoadedFor = current.evidence_packet_id;
    renderAnswer(current, latest.query, resumeMeta);
    setBusy(true, "Checking whether the session is still active…");
    try {
      await ensureDemoToken();
      const tokenRenewed = activeSessionId !== session.id;
      if (tokenRenewed) {
        setStatus("The saved answer was restored, but the renewed demo token starts a new server session.", "working");
        return;
      }
      const state = await loadSessionState(session.id);
      const decision = browserState.publicResumeDecision(session, {
        tokenRenewed,
        serverLive: state?.live,
      });
      if (decision.continueSession) {
        activeSessionLive = true;
        clearStatus();
      } else {
        detachActiveSession();
        setStatus("The saved answer was restored, but its server context has expired. Start a new session before asking another question.", "error");
      }
    } catch (error) {
      detachActiveSession();
      setStatus(`The saved answer was restored, but session liveness could not be verified: ${error.message}`, "error");
    } finally {
      setBusy(false);
      renderSessionState();
      renderHistory();
    }
    return;
  }
  if (!lastTurn?.answer_id) {
    setBusy(true, "Checking whether the session is still active…");
    try {
      const state = await loadSessionState(session.id);
      if (state?.live) {
        activeSessionLive = true;
        setStatus("Session resumed. Ask a follow-up question.", "working");
      } else {
        detachActiveSession();
        setStatus("This session's server context has expired. Start a new session.", "error");
      }
    } finally {
      setBusy(false);
    }
    renderSessionState();
    renderHistory();
    queryInput.focus();
    return;
  }
  setBusy(true, "Loading the session's latest answer…");
  const [stateResult, answerResult] = await Promise.allSettled([
    requestJson(`/api/sessions/${session.id}`),
    requestJson(`/api/answers/${lastTurn.answer_id}`),
  ]);
  try {
    if (answerResult.status === "rejected") {
      throw answerResult.reason;
    }
    current = answerResult.value;
    packetLoadedFor = null;
    renderAnswer(current, lastTurn.query, resumeMeta);
    if (stateResult.status === "fulfilled" && stateResult.value.live) {
      activeSessionLive = true;
      clearStatus();
    } else {
      detachActiveSession();
      const message = stateResult.status === "rejected"
        ? "The stored answer was loaded, but server session status could not be verified. Start a new session before asking another question."
        : "The stored answer was loaded, but its server conversation context has expired. Start a new session before asking another question.";
      setStatus(message, "error");
    }
    renderSessionState();
    renderHistory();
  } catch (error) {
    clearResult();
    if (error.status === 404) {
      removeSession(session.id);
      setStatus("This local history entry no longer exists on the server, so it was removed.", "error");
    } else {
      detachActiveSession();
      setStatus(`The session's stored answer could not be loaded: ${error.message}`, "error");
    }
    renderSessionState();
    renderHistory();
  } finally {
    setBusy(false);
    queryInput.focus();
  }
}

async function restoreActiveSession() {
  const session = activeSessionId ? findSession(activeSessionId) : null;
  if (session) await resumeSession(session);
}

async function loadSessionState(sessionId) {
  try {
    return await requestJson(`/api/sessions/${sessionId}`);
  } catch {
    return null;
  }
}

function detachActiveSession() {
  activeSessionId = null;
  activeSessionLive = false;
  sessionStore.removeItem(ACTIVE_KEY);
}

function removeSession(sessionId) {
  sessions = sessions.filter((entry) => entry.id !== sessionId);
  saveSessions();
  if (activeSessionId === sessionId) detachActiveSession();
}

function clearResult() {
  current = null;
  currentPacket = null;
  currentMeta = null;
  packetLoadedFor = null;
  result.hidden = true;
  progressPanel.hidden = true;
  hero.hidden = false;
  shell.classList.remove("has-answer");
}

function truncate(text, max) {
  const characters = Array.from(String(text));
  return characters.length > max ? `${characters.slice(0, max - 1).join("")}…` : characters.join("");
}

function relativeTime(timestamp) {
  const seconds = Math.round((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

function clockTime(timestamp) {
  if (!timestamp) return "";
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

/* ---------- details visibility ---------- */

function applyDetailsVisibility() {
  detailsSection.hidden = !detailsToggle.checked || !current;
}

async function loadEvidencePacket() {
  if (!current) return;
  const packetId = current.evidence_packet_id;
  try {
    const packet = appMode === "public"
      ? current.evidence_packet
      : await requestJson(`/api/evidence-packets/${packetId}`);
    if (!packet) throw new Error("The answer does not contain packet details.");
    if (current.evidence_packet_id !== packetId) return;
    packetLoadedFor = packetId;
    currentPacket = packet;
    renderPlan(packet);
    renderEvidencePacket(packet);
    renderCitedEvidence(current);
    renderTrace(current);
  } catch (error) {
    const message = `Could not load the persisted evidence packet: ${error.message}`;
    setPanel("#d-plan", emptyMessage(message));
    setPanel("#d-evidence", emptyMessage(message));
  }
}

async function requestJson(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (appMode === "public" && !url.startsWith("/api/demo/session")) {
    const token = sessionStorage.getItem(DEMO_TOKEN_KEY);
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }
  options = { ...options, headers };
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body?.error?.message || `Request failed (${response.status}).`);
    error.status = response.status;
    error.code = body?.error?.code;
    if (response.status === 401 && appMode === "public") {
      sessionStorage.removeItem(DEMO_TOKEN_KEY);
      sessionStorage.removeItem(DEMO_TOKEN_EXP_KEY);
      detachActiveSession();
      renderSessionState();
    }
    throw error;
  }
  return body;
}

/* ---------- live stages ---------- */

const STAGE_LABELS = {
  loading_embedding_model: "Loading the embedding model",
  planning: "Planning the search",
  keyword_search: "Running keyword search",
  semantic_search: "Running semantic search",
  answer_generation: "Generating the answer",
};
const DEFAULT_STAGE_PHASES = [
  "planning",
  "keyword_search",
  "semantic_search",
  "answer_generation",
];

function defaultStages() {
  return DEFAULT_STAGE_PHASES.map((phase) => ({
    phase,
    label: STAGE_LABELS[phase],
    status: "queued",
    seconds: null,
  }));
}

function beginQuestion(query) {
  const session = activeSessionId ? findSession(activeSessionId) : null;
  const turnIndex = (session?.turns.length || 0) + 1;
  hero.hidden = true;
  current = null;
  currentPacket = null;
  currentMeta = null;
  packetLoadedFor = null;
  shell.classList.remove("has-answer");
  result.hidden = false;
  questionIndexLabel.textContent = `Question ${String(turnIndex).padStart(2, "0")}`;
  questionMetaDetail.textContent = "asking now";
  askedQuery.textContent = query;
  receiptBox.hidden = true;
  answerNotice.hidden = true;
  answerCard.hidden = true;
  evidenceCited.hidden = true;
  searchedPanel.hidden = true;
  detailsSection.hidden = true;
  progressPanel.hidden = false;
  clearStatus();
}

function markPhase(request, phase, now) {
  if (!phase || request.currentPhase === phase) return;
  let index = request.stages.findIndex((stage) => stage.phase === phase);
  if (index === -1) {
    // A cold start reports a stage that is not part of the standard four.
    request.stages.unshift({
      phase,
      label: STAGE_LABELS[phase] || phase,
      status: "queued",
      seconds: null,
    });
    index = 0;
    if (request.activeIndex >= 0) request.activeIndex += 1;
  }
  request.stages.forEach((stage, position) => {
    if (position >= index || stage.status === "done") return;
    stage.status = "done";
    if (position === request.activeIndex && request.activeStartedAt !== null) {
      stage.seconds = (now - request.activeStartedAt) / 1000;
    }
  });
  request.stages[index].status = "active";
  request.activeIndex = index;
  request.activeStartedAt = now;
  request.currentPhase = phase;
}

function completeStages(request) {
  const now = Date.now();
  request.stages.forEach((stage, position) => {
    if (position === request.activeIndex && request.activeStartedAt !== null) {
      stage.seconds = (now - request.activeStartedAt) / 1000;
    }
    stage.status = "done";
  });
}

function startRequestFeedback(request) {
  cancelRequestButton.hidden = false;
  cancelRequestButton.disabled = false;
  renderRequestFeedback(request);
  request.elapsedTimer = window.setInterval(() => renderRequestFeedback(request), 250);
  request.progressTimer = window.setInterval(async () => {
    try {
      const progress = await requestJson(`/api/asks/${request.requestId}/progress`);
      if (activeRequest === request && !request.cancelled) {
        request.progress = progress;
        markPhase(request, progress.phase, Date.now());
        renderRequestFeedback(request);
      }
    } catch {
      // Preserve the generic panel when a server cannot provide telemetry.
    }
  }, 1000);
}

function stopRequestFeedback(request) {
  window.clearInterval(request.elapsedTimer);
  window.clearInterval(request.progressTimer);
  if (activeRequest !== request) return;
  // Clear the active request first so the header chip drops its live state.
  activeRequest = null;
  cancelRequestButton.hidden = true;
  progressPanel.hidden = true;
  renderTopChip();
}

function renderRequestFeedback(request) {
  if (activeRequest !== request || request.cancelled) return;
  const now = Date.now();
  const elapsed = (now - request.startedAt) / 1000;
  progressTitle.textContent = `Working — ${elapsed.toFixed(1)}s elapsed`;
  const done = request.stages.filter((stage) => stage.status === "done").length;
  const activeIndex = request.stages.findIndex((stage) => stage.status === "active");
  progressStage.textContent = activeIndex >= 0
    ? `stage ${activeIndex + 1} of ${request.stages.length}`
    : `${done} of ${request.stages.length} stages recorded`;
  renderStages(request, now);
  renderTopChip();
}

function renderStages(request, now) {
  const fragment = document.createDocumentFragment();
  request.stages.forEach((stage, position) => {
    const item = document.createElement("li");
    item.className = `stage ${stage.status}`;

    const mark = document.createElement("span");
    mark.className = "stage-mark";
    if (stage.status === "done") mark.textContent = "✓";

    const body = document.createElement("div");
    body.className = "stage-body";
    const line = document.createElement("div");
    line.className = "stage-line";
    const name = document.createElement("span");
    name.className = "stage-name";
    name.textContent = stage.label;
    const time = document.createElement("span");
    time.className = "stage-time";
    if (stage.status === "active" && request.activeStartedAt !== null && position === request.activeIndex) {
      time.textContent = `${((now - request.activeStartedAt) / 1000).toFixed(1)}s…`;
    } else if (stage.status === "done") {
      time.textContent = stage.seconds === null ? "done" : `${stage.seconds.toFixed(1)}s`;
    } else {
      time.textContent = "queued";
    }
    line.append(name, time);
    body.append(line);
    item.append(mark, body);
    fragment.append(item);
  });
  stageList.replaceChildren(fragment);
}

/* ---------- answer ---------- */

function renderAnswer(payload, queryText, meta = {}) {
  currentMeta = meta;
  currentPacket = appMode === "public" ? payload.evidence_packet || null : null;
  hero.hidden = true;
  progressPanel.hidden = true;
  shell.classList.add("has-answer");
  result.hidden = false;

  const turnIndex = meta.turnIndex || 1;
  questionIndexLabel.textContent = `Question ${String(turnIndex).padStart(2, "0")}`;
  questionMetaDetail.textContent = describeRun(payload, meta);
  askedQuery.textContent = queryText || "";

  const answerStatus = payload.answer_status || "answered";
  const isFailure = answerStatus !== "answered";
  answerCard.hidden = false;
  answerCard.classList.toggle("answer-failure", isFailure);
  answerState.hidden = !isFailure;
  answerState.textContent = answerStatus === "validation_failed"
    ? "Answer generation failed validation"
    : "Insufficient source evidence";

  renderReceipt(payload, meta, isFailure);
  renderNotice(payload, answerStatus);
  renderProse(payload);
  renderCitedEvidence(payload);
  renderSearched(payload, answerStatus);
  renderCitations(payload.citations || []);
  renderLimitations(payload.limitations || []);
  renderCoverage(payload.coverage || {});
  renderTrace(payload);
  if (currentPacket) {
    // Public mode ships the packet with the answer; nothing else to fetch.
    packetLoadedFor = payload.evidence_packet_id;
    renderPlan(currentPacket);
    renderEvidencePacket(currentPacket);
  } else {
    setPanel("#d-plan", emptyMessage("Loading persisted evidence packet…"));
    setPanel("#d-evidence", emptyMessage("Loading persisted evidence packet…"));
    // The quotes on the cited-evidence cards come from the packet, so it is
    // loaded for every answer rather than only when the trace is open.
    loadEvidencePacket();
  }
  applyDetailsVisibility();
  result.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
}

function describeRun(payload, meta) {
  const parts = [];
  if (meta.at) parts.push(clockTime(meta.at));
  if (typeof meta.elapsedSeconds === "number") {
    parts.push(`answered in ${meta.elapsedSeconds.toFixed(1)}s`);
  }
  if (payload.search_run_id) parts.push(`run ${payload.search_run_id}`);
  return parts.join(" · ");
}

function renderReceipt(payload, meta, isFailure = false) {
  const coverage = payload.coverage || {};
  const fragment = document.createDocumentFragment();

  const head = document.createElement("div");
  head.className = "receipt-head";
  const dot = document.createElement("span");
  dot.className = "receipt-dot";
  const title = document.createElement("strong");
  const verb = isFailure ? "Finished" : "Answered";
  title.textContent = typeof meta.elapsedSeconds === "number"
    ? `${verb} in ${meta.elapsedSeconds.toFixed(1)}s`
    : "Restored from this browser's history";
  const summary = document.createElement("span");
  summary.className = "meta";
  const scanned = coverage.raw_result_count;
  const kept = coverage.evidence_count;
  const bits = [];
  if (typeof scanned === "number") bits.push(`${formatCount(scanned)} results scanned`);
  if (typeof kept === "number") bits.push(`${formatCount(kept)} kept as evidence`);
  if (typeof coverage.evidence_token_count === "number") {
    bits.push(`${formatCount(coverage.evidence_token_count)} tokens`);
  }
  summary.textContent = bits.join(" · ");
  head.append(dot, title, summary);
  fragment.append(head);

  const stages = (meta.stages || []).filter((stage) => stage.status === "done");
  if (stages.length) {
    const grid = document.createElement("div");
    grid.className = "receipt-stages";
    stages.forEach((stage) => {
      const cell = document.createElement("div");
      const bar = document.createElement("div");
      bar.className = "receipt-stage-bar";
      const name = document.createElement("div");
      name.className = "receipt-stage-name";
      name.textContent = stage.label;
      const time = document.createElement("div");
      time.className = "receipt-stage-time";
      time.textContent = stage.seconds === null ? "recorded" : `${stage.seconds.toFixed(1)}s`;
      cell.append(bar, name, time);
      grid.append(cell);
    });
    fragment.append(grid);
    const note = document.createElement("p");
    note.className = "receipt-note";
    note.textContent = "Stage times are measured in this browser, so they are accurate to about a second.";
    fragment.append(note);
  }

  receiptBox.replaceChildren(fragment);
  receiptBox.hidden = false;
}

function renderNotice(payload, answerStatus) {
  if (answerStatus === "insufficient_evidence") {
    setNotice(
      "hard",
      "No chunk-backed evidence found",
      "Every planned stage ran, but nothing in your indexed files supports an answer to this. Rather than write something plausible, the agent stops here.",
    );
    return;
  }
  if (answerStatus === "validation_failed") {
    setNotice(
      "hard",
      "Answer generation failed validation",
      "The generated answer did not satisfy the citation contract, so it was refused instead of shown. The recorded limitations below say which check failed.",
    );
    return;
  }
  answerNotice.hidden = true;
}

function setNotice(kind, title, body) {
  const mark = document.createElement("span");
  mark.className = "notice-mark";
  mark.textContent = "!";
  const block = document.createElement("div");
  block.className = "notice-text";
  const heading = document.createElement("div");
  heading.className = "notice-title";
  heading.textContent = title;
  const text = document.createElement("div");
  text.className = "notice-body";
  text.textContent = body;
  block.append(heading, text);
  answerNotice.className = kind === "hard" ? "notice hard" : "notice";
  answerNotice.replaceChildren(mark, block);
  answerNotice.hidden = false;
}

const CITATION_MARKER = /\[(E\d+)\]/g;

function renderProse(payload) {
  answerRoot.replaceChildren();
  const paragraphs = String(payload.answer_body).split(/\n+/).filter((p) => p.trim());
  (paragraphs.length ? paragraphs : ["No answer text was returned."]).forEach((text) => {
    const p = document.createElement("p");
    p.dir = "auto";
    let cursor = 0;
    const value = text.trim();
    for (const match of value.matchAll(CITATION_MARKER)) {
      const before = value.slice(cursor, match.index).replace(/\s+$/, "");
      if (before) p.append(document.createTextNode(before));
      p.append(citationChip(match[1]));
      cursor = match.index + match[0].length;
    }
    const tail = value.slice(cursor);
    if (tail) p.append(document.createTextNode(tail));
    answerRoot.append(p);
  });
}

function citationChip(citationId) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "cite";
  chip.textContent = citationId;
  chip.title = `Show evidence ${citationId}`;
  chip.addEventListener("click", () => revealEvidence(citationId));
  return chip;
}

function revealEvidence(citationId) {
  const card = document.querySelector(`[data-evidence="${citationId}"]`);
  if (!card) {
    setStatus(`Evidence ${citationId} is not available in this answer.`, "error");
    return;
  }
  card.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
  card.classList.add("flash");
  window.setTimeout(() => card.classList.remove("flash"), 1600);
}

function renderCitedEvidence(payload) {
  const references = payload.references || [];
  if (!references.length) {
    evidenceCited.hidden = true;
    return;
  }
  const byId = new Map();
  (payload.citations || []).forEach((citation) => {
    if (!byId.has(citation.citation_id)) byId.set(citation.citation_id, citation);
  });
  const items = currentPacket?.evidence || [];

  const head = document.createElement("div");
  head.className = "section-head";
  const label = document.createElement("span");
  label.className = "section-label";
  label.textContent = "Evidence";
  const meta = document.createElement("span");
  meta.className = "meta";
  const courses = [...new Set(references.map((item) => item.course))];
  const files = [...new Set(references.map((item) => item.file_path))];
  meta.textContent = [
    `${references.length} cited chunk${references.length === 1 ? "" : "s"}`,
    courses.length === 1 ? courses[0] : `${courses.length} courses`,
    files.length === 1 ? fileName(files[0]) : `${files.length} files`,
  ].join(" · ");
  head.append(label, meta);

  const grid = document.createElement("div");
  grid.className = "evidence-grid";
  references.forEach((reference) => {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.dataset.evidence = reference.citation_id;

    const cardHead = document.createElement("div");
    cardHead.className = "evidence-card-head";
    const id = document.createElement("span");
    id.className = "evidence-id";
    id.textContent = reference.citation_id;
    const where = document.createElement("span");
    where.className = "evidence-where";
    where.textContent = `${reference.location_label} · ${fileName(reference.file_path)}`;
    where.title = `${reference.course} · ${reference.file_path}`;
    cardHead.append(id, where);

    const quote = document.createElement("div");
    quote.className = "evidence-quote";
    const citation = byId.get(reference.citation_id);
    const item = citation ? items[citation.evidence_index - 1] : null;
    quote.textContent = item?.text
      ? `“${truncate(item.text, 220)}”`
      : `${reference.course} · ${reference.source_type}`;

    card.append(cardHead, quote);
    grid.append(card);
  });

  evidenceCited.replaceChildren(head, grid);
  evidenceCited.hidden = false;
}

function renderSearched(payload, answerStatus) {
  if (answerStatus !== "insufficient_evidence") {
    searchedPanel.hidden = true;
    return;
  }
  const coverage = payload.coverage || {};
  const head = document.createElement("div");
  head.className = "section-head";
  const label = document.createElement("span");
  label.className = "section-label";
  label.textContent = "What was searched";
  head.append(label);
  searchedPanel.replaceChildren(head, coverageRows(coverage));
  searchedPanel.hidden = false;
}

function coverageRows(coverage) {
  const hits = new Set([
    ...(coverage.courses_with_chunk_hits || []),
    ...(coverage.indexes_with_chunk_hits || []),
  ]);
  const sources = [
    ...(coverage.searched_courses || []),
    ...(coverage.searched_indexes || []),
  ];
  const rows = document.createElement("div");
  rows.className = "rows";
  if (!sources.length) {
    rows.append(emptyMessage("No sources were recorded for this run."));
    return rows;
  }
  sources.forEach((source) => {
    const row = document.createElement("div");
    row.className = "row";
    const name = document.createElement("span");
    name.className = "row-name";
    name.textContent = source;
    const value = document.createElement("span");
    value.className = hits.has(source) ? "row-value hit" : "row-value miss";
    value.textContent = hits.has(source) ? "evidence found" : "no hits";
    row.append(name, value);
    rows.append(row);
  });
  return rows;
}

/* ---------- detail panels ---------- */

function renderCitations(citations) {
  setPanelMeta("#d-citations-meta", citations.length ? `${citations.length} cited` : "");
  if (!citations.length) {
    setPanel("#d-citations", emptyMessage("No citations were needed for this answer."));
    return;
  }
  const table = buildTable(
    ["ID", "Course", "File", "Type", "Location", "Evidence #", "File ID", "Chunk ID"],
    citations.map((c) => [
      c.citation_id,
      c.course,
      c.file_path,
      c.source_type,
      c.location_label,
      c.evidence_index,
      c.file_id,
      c.chunk_id,
    ]),
  );
  setPanel("#d-citations", table);
}

function renderLimitations(limitations) {
  limitations = uniqueStrings(limitations);
  setPanelMeta(
    "#d-limitations-meta",
    limitations.length ? `${limitations.length} recorded` : "none recorded",
  );
  if (!limitations.length) {
    setPanel("#d-limitations", emptyMessage("No limitations reported."));
    return;
  }
  setPanel("#d-limitations", stringList(limitations));
}

function renderCoverage(coverage) {
  const projected = { ...coverage };
  const unshownWeaknesses = additionalWeaknesses(projected.weaknesses || []);
  delete projected.weaknesses;
  delete projected.searched_courses;
  delete projected.searched_indexes;
  delete projected.courses_with_chunk_hits;
  delete projected.indexes_with_chunk_hits;
  delete projected.courses_without_chunk_hits;
  delete projected.indexes_without_chunk_hits;
  if (unshownWeaknesses.length) projected.additional_weaknesses = unshownWeaknesses;
  const hits = (coverage.courses_with_chunk_hits || []).length
    + (coverage.indexes_with_chunk_hits || []).length;
  const planned = (coverage.searched_courses || []).length
    + (coverage.searched_indexes || []).length;
  setPanelMeta("#d-coverage-meta", planned ? `${planned} planned / ${hits} productive` : "");
  const fragment = document.createDocumentFragment();
  fragment.append(subheading("Sources searched"), coverageRows(coverage));
  fragment.append(subheading("Recorded counts"), definitionGrid(projected));
  setPanel("#d-coverage", fragment);
}

function renderPlan(packet) {
  const settings = packet.retrieval_settings || {};
  setPanelMeta(
    "#d-plan-meta",
    settings.keyword_top_k === undefined
      ? ""
      : `top-k ${settings.keyword_top_k}/${settings.semantic_top_k}/${settings.metadata_top_k} · RRF ${settings.rrf_k}`,
  );
  const fragment = document.createDocumentFragment();
  fragment.append(
    subheading("Interpreted intent"),
    paragraph(packet.interpreted_intent || "Not recorded."),
    subheading("Query plan"),
    definitionGrid(packet.query_plan || {}),
    subheading("Retrieval settings"),
    definitionGrid(settings),
    subheading("Searched"),
    definitionGrid(packet.searched || {}),
  );
  setPanel("#d-plan", fragment);
}

function renderEvidencePacket(packet) {
  const fragment = document.createDocumentFragment();
  const items = packet.evidence || [];
  const cited = new Set((current?.citations || []).map((c) => c.evidence_index));
  setPanelMeta(
    "#d-evidence-meta",
    `${items.length} item${items.length === 1 ? "" : "s"} · ${cited.size} cited`,
  );

  const unshownWeaknesses = additionalWeaknesses(packet.weaknesses || []);
  if (unshownWeaknesses.length) {
    fragment.append(subheading("Additional weaknesses"), stringList(unshownWeaknesses));
  }
  if (packet.answer_constraints?.length) {
    fragment.append(subheading("Answer constraints"), stringList(packet.answer_constraints));
  }

  fragment.append(subheading(`Evidence items (${items.length})`));
  if (!items.length) {
    fragment.append(emptyMessage("The packet contains no evidence items."));
  }
  items.forEach((item, index) => {
    const article = document.createElement("article");
    article.className = "evidence-item";

    const head = document.createElement("div");
    head.className = "evidence-head";
    const title = document.createElement("strong");
    title.textContent = cited.has(index + 1)
      ? `E${index + 1} · ${item.course}`
      : `E${index + 1} · not cited · ${item.course}`;
    head.append(
      title,
      badge(item.retrieval_method),
      badge(`rank ${item.rank}`),
      badge(`score ${formatNumber(item.score)}`),
      badge(`${item.token_count} tokens`),
    );
    article.append(head);

    const path = document.createElement("small");
    path.className = "evidence-path";
    path.textContent = `${item.file} · ${item.source_type} · ${item.location?.label || "location unavailable"} · file ${item.file_id} · chunk ${item.chunk_id}`;
    article.append(path);

    const text = document.createElement("p");
    text.className = "evidence-text";
    text.textContent = item.text;
    article.append(text);

    if (item.contributions?.length) {
      const contributions = document.createElement("div");
      contributions.className = "contributions";
      item.contributions.forEach((entry) => {
        contributions.append(badge(Object.entries(entry).map(([k, v]) => `${k}: ${formatValue(v)}`).join(" · "), "soft"));
      });
      article.append(contributions);
    }
    fragment.append(article);
  });
  setPanel("#d-evidence", fragment);
}

function renderTrace(payload) {
  const fragment = document.createDocumentFragment();
  const settings = currentPacket?.retrieval_settings || {};
  setPanelMeta("#d-trace-meta", settings.embedding_model || "");

  const lines = [];
  const elapsed = typeof currentMeta?.elapsedSeconds === "number"
    ? `${currentMeta.elapsedSeconds.toFixed(2)}s`
    : "restored";
  lines.push([
    ["b", "POST"],
    ["t", ` /api/ask · ${elapsed} · run ${payload.search_run_id ?? "—"}`],
  ]);
  (currentMeta?.stages || []).forEach((stage) => {
    if (stage.status !== "done") return;
    const seconds = stage.seconds === null ? "recorded" : `${stage.seconds.toFixed(2)}s`;
    lines.push([["t", `  phase ${stage.phase} · ${seconds}`]]);
  });
  const coverage = payload.coverage || {};
  if (typeof coverage.raw_result_count === "number") {
    lines.push([["t", `  raw hits ${coverage.raw_result_count} · fused ${coverage.fused_candidate_count} · packet ${coverage.evidence_count} · tokens ${coverage.evidence_token_count}`]]);
  }
  if (settings.embedding_model) {
    lines.push([["t", `  embed ${settings.embedding_model} · rrf k=${settings.rrf_k} · queries ${settings.semantic_query_limit}`]]);
  }
  lines.push([["t", `  answer ${payload.answer_id ?? "—"} · packet ${payload.evidence_packet_id ?? "—"} · citations ${(payload.citations || []).length} · limitations ${(payload.limitations || []).length}`]]);

  const block = document.createElement("div");
  block.className = "trace-lines";
  lines.forEach((parts) => {
    const row = document.createElement("div");
    parts.forEach(([kind, text]) => {
      if (kind === "b") {
        const strong = document.createElement("b");
        strong.textContent = text;
        row.append(strong);
      } else {
        row.append(document.createTextNode(text));
      }
    });
    block.append(row);
  });
  fragment.append(block);

  if (appMode === "public") {
    fragment.append(paragraph("Historical numeric-ID lookup routes are disabled in public mode; this response already contains the complete packet detail."));
    setPanel("#d-trace", fragment);
    return;
  }
  const links = document.createElement("div");
  links.className = "api-links";
  [
    ["Answer JSON", payload.answer_id, (id) => `/api/answers/${id}`],
    ["Coverage JSON", payload.search_run_id, (id) => `/api/search-runs/${id}/coverage`],
    ["Evidence packet JSON", payload.evidence_packet_id, (id) => `/api/evidence-packets/${id}`],
  ].forEach(([label, id, makeHref]) => {
    if (!Number.isInteger(id) || id <= 0) return;
    const a = document.createElement("a");
    a.href = makeHref(id);
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = label;
    links.append(a);
  });
  fragment.append(links);
  setPanel("#d-trace", fragment);
}

function showRequestFailure(query, error) {
  hero.hidden = true;
  result.hidden = false;
  progressPanel.hidden = true;
  shell.classList.add("has-answer");
  askedQuery.textContent = query;
  questionMetaDetail.textContent = error.code ? `failed · ${error.code}` : "failed";
  receiptBox.hidden = true;
  answerCard.hidden = true;
  evidenceCited.hidden = true;
  searchedPanel.hidden = true;
  detailsSection.hidden = true;
  setNotice("hard", "This question could not be answered", error.message);
}

/* ---------- generic builders ---------- */

function definitionGrid(object) {
  const entries = Object.entries(object || {});
  if (!entries.length) return emptyMessage("Nothing recorded.");
  const grid = document.createElement("dl");
  grid.className = "kv-grid";
  entries.forEach(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = prettyKey(key);
    const dd = document.createElement("dd");
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      dd.append(definitionGrid(value));
    } else {
      dd.textContent = formatValue(value);
    }
    grid.append(dt, dd);
  });
  return grid;
}

function buildTable(headers, rows) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headRow.append(th);
  });
  thead.append(headRow);
  const tbody = document.createElement("tbody");
  rows.forEach((cells) => {
    const tr = document.createElement("tr");
    cells.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = formatValue(cell);
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(thead, tbody);
  wrap.append(table);
  return wrap;
}

function stringList(items) {
  const ul = document.createElement("ul");
  ul.className = "plain-list";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = formatValue(item);
    ul.append(li);
  });
  return ul;
}

function uniqueStrings(items) {
  return [...new Set((items || []).map((item) => String(item).trim()).filter(Boolean))];
}

function additionalWeaknesses(weaknesses) {
  const displayedLimitations = new Set(uniqueStrings(current?.limitations || []));
  return uniqueStrings(weaknesses)
    .filter((weakness) => !displayedLimitations.has(weakness));
}

function badge(text, variant) {
  const span = document.createElement("span");
  span.className = variant ? `badge ${variant}` : "badge";
  span.textContent = text;
  return span;
}

function subheading(text) {
  const h = document.createElement("h3");
  h.className = "subheading";
  h.textContent = text;
  return h;
}

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function emptyMessage(text) {
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = text;
  return p;
}

function setPanel(selector, node) {
  const root = document.querySelector(selector);
  if (root) root.replaceChildren(node);
}

function setPanelMeta(selector, text) {
  const root = document.querySelector(selector);
  if (root) root.textContent = text || "";
}

function prettyKey(key) {
  return key.replaceAll("_", " ");
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.length ? value.map(formatValue).join(" · ") : "none";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function formatNumber(value) {
  if (typeof value !== "number") return String(value);
  return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatCount(value) {
  return typeof value === "number" ? value.toLocaleString() : String(value ?? "—");
}

function fileName(path) {
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function setBusy(busy, message = "") {
  askButton.disabled = busy;
  askButton.querySelector(".button-label").textContent = busy ? "Working…" : "Ask";
  newSessionButton.disabled = busy;
  clearHistoryButton.disabled = busy;
  if (!busy) {
    cancelRequestButton.hidden = true;
    cancelRequestButton.disabled = false;
  }
  if (message) setStatus(message, "working");
  renderHistory();
}

function setStatus(message, kind) {
  statusBox.hidden = false;
  statusBox.textContent = message;
  statusBox.className = `status ${kind || ""}`;
}

function clearStatus() {
  statusBox.hidden = true;
  statusBox.textContent = "";
}
