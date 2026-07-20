const form = document.querySelector("#ask-form");
const queryInput = document.querySelector("#query");
const askButton = document.querySelector("#ask-button");
const statusBox = document.querySelector("#status");
const result = document.querySelector("#result");
const detailsToggle = document.querySelector("#details-toggle");
const detailsSection = document.querySelector("#details");
const historySection = document.querySelector("#history");
const historyList = document.querySelector("#history-list");
const activeSessionLabel = document.querySelector("#active-session-label");
const newSessionButton = document.querySelector("#new-session");
const cancelRequestButton = document.querySelector("#cancel-request");
const settingsButton = document.querySelector("#settings-button");
const settingsDialog = document.querySelector("#settings-dialog");
const settingsForm = document.querySelector("#settings-form");
const settingsFields = document.querySelector("#settings-fields");
const settingsStatus = document.querySelector("#settings-status");
const settingsSaveButton = document.querySelector("#settings-save");
const settingsResetButton = document.querySelector("#settings-reset");
const settingsCloseButton = document.querySelector("#settings-close");

const SESSIONS_KEY = "uni-rag-sessions";
const ACTIVE_KEY = "uni-rag-active-session";
const DETAILS_KEY = "uni-rag-details";

let current = null;
let packetLoadedFor = null;
let sessions = loadSessions();
let activeSessionId = localStorage.getItem(ACTIVE_KEY);
if (activeSessionId && !findSession(activeSessionId)) activeSessionId = null;
let activeSessionLive = activeSessionId ? null : false;
let activeRequest = null;

detailsToggle.checked = localStorage.getItem(DETAILS_KEY) === "1";
applyDetailsVisibility();
renderSessionState();
renderHistory();
restoreActiveSession();

detailsToggle.addEventListener("change", () => {
  localStorage.setItem(DETAILS_KEY, detailsToggle.checked ? "1" : "0");
  applyDetailsVisibility();
});

newSessionButton.addEventListener("click", () => {
  activeSessionId = null;
  activeSessionLive = false;
  current = null;
  packetLoadedFor = null;
  localStorage.removeItem(ACTIVE_KEY);
  clearResult();
  clearStatus();
  queryInput.value = "";
  renderSessionState();
  renderHistory();
  queryInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (activeSessionLive === null) {
    setStatus("Wait for the active-session check to finish before asking.", "working");
    return;
  }
  const query = queryInput.value.trim();
  if (!query) return;
  const requestId = generateRequestId();
  const controller = new AbortController();
  const request = {
    requestId,
    controller,
    cancelled: false,
    progressTimer: null,
    elapsedTimer: null,
    startedAt: Date.now(),
  };
  activeRequest = request;
  setBusy(true, "Searching your indexed materials…");
  startRequestFeedback(request);
  const sessionId = activeSessionId && activeSessionLive ? activeSessionId : generateSessionId();
  try {
    current = await requestJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: sessionId, request_id: requestId }),
      signal: controller.signal,
    });
    packetLoadedFor = null;
    activeSessionId = sessionId;
    activeSessionLive = true;
    localStorage.setItem(ACTIVE_KEY, sessionId);
    recordTurn(sessionId, query, current.answer_id);
    renderSessionState();
    renderHistory();
    renderAnswer(current, query);
    queryInput.value = "";
    clearStatus();
    applyDetailsVisibility();
  } catch (error) {
    if (!request.cancelled) setStatus(error.message, "error");
  } finally {
    stopRequestFeedback(request);
    if (activeRequest === request) activeRequest = null;
    setBusy(false);
  }
});

cancelRequestButton.addEventListener("click", async () => {
  const request = activeRequest;
  if (!request) return;
  cancelRequestButton.disabled = true;
  try {
    const result = await requestJson(`/api/asks/${request.requestId}/cancel`, {
      method: "POST",
    });
    if (result.cancelled) {
      request.cancelled = true;
      setStatus("Request cancelled. Any in-flight work will finish without saving an answer.", "working");
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
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});

/* ---------- retrieval settings dialog ---------- */

const SETTING_LABELS = {
  embedding_model: "Embedding model",
  keyword_top_k: "Keyword results (top k)",
  semantic_top_k: "Semantic results (top k)",
  metadata_top_k: "Metadata results (top k)",
  final_top_k: "Final evidence items (top k)",
  rrf_k: "RRF rank constant",
  semantic_query_limit: "Semantic queries per ask",
  filename_fuzzy_threshold: "Filename fuzzy threshold",
  path_fuzzy_threshold: "Path fuzzy threshold",
  evidence_max_tokens: "Evidence token budget",
  query_plan_min_confidence: "Minimum plan confidence",
};
const FLOAT_SETTINGS = new Set(["query_plan_min_confidence"]);
let settingsPayload = null;

settingsButton.addEventListener("click", async () => {
  settingsDialog.showModal();
  clearSettingsStatus();
  settingsFields.replaceChildren(emptyMessage("Loading current settings…"));
  try {
    settingsPayload = await requestJson("/api/settings");
    renderSettingsForm(settingsPayload);
  } catch (error) {
    settingsFields.replaceChildren(
      emptyMessage(`Could not load settings: ${error.message}`),
    );
  }
});

settingsCloseButton.addEventListener("click", () => settingsDialog.close());

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!settingsPayload) return;
  const changes = {};
  let invalid = null;
  Object.keys(SETTING_LABELS).forEach((name) => {
    const input = settingsForm.querySelector(`[name="${name}"]`);
    if (!input) return;
    const raw = input.value.trim();
    if (raw === "") {
      changes[name] = null;
      return;
    }
    if (name === "embedding_model") {
      changes[name] = raw;
      return;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      invalid = invalid || `${SETTING_LABELS[name]} must be a number.`;
      return;
    }
    changes[name] = value;
  });
  if (invalid) {
    setSettingsStatus(invalid, "error");
    return;
  }
  await submitSettings(changes, "Settings saved. They apply from your next question.");
});

settingsResetButton.addEventListener("click", async () => {
  if (!settingsPayload) return;
  const changes = {};
  Object.keys(SETTING_LABELS).forEach((name) => {
    changes[name] = null;
  });
  await submitSettings(changes, "All settings now follow the server configuration.");
});

async function submitSettings(changes, successMessage) {
  settingsSaveButton.disabled = true;
  settingsResetButton.disabled = true;
  try {
    settingsPayload = await requestJson("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    });
    renderSettingsForm(settingsPayload);
    setSettingsStatus(successMessage, "ok");
  } catch (error) {
    setSettingsStatus(error.message, "error");
  } finally {
    settingsSaveButton.disabled = false;
    settingsResetButton.disabled = false;
  }
}

function renderSettingsForm(payload) {
  const fragment = document.createDocumentFragment();
  fragment.append(buildEmbeddingModelField(payload));
  Object.keys(SETTING_LABELS)
    .filter((name) => name !== "embedding_model")
    .forEach((name) => fragment.append(buildNumericField(payload, name)));
  settingsFields.replaceChildren(fragment);
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
    ? `Server default (${configured})`
    : "Server default (not set)";
  select.append(fallback);
  (payload.embedding_model_profiles || []).forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.model_name;
    option.textContent = `${profile.model_name} · ${profile.provider} · ${profile.dimension}d`;
    select.append(option);
  });
  select.value = payload.overrides.embedding_model || "";
  field.append(select);
  return field;
}

function buildNumericField(payload, name) {
  const field = settingsField(name);
  const input = document.createElement("input");
  input.type = "number";
  input.name = name;
  input.id = `setting-${name}`;
  const limits = payload.limits?.[name];
  if (limits) {
    input.min = limits.min;
    input.max = limits.max;
  }
  input.step = FLOAT_SETTINGS.has(name) ? "0.05" : "1";
  input.placeholder = `default: ${formatValue(payload.defaults[name])}`;
  const override = payload.overrides[name];
  input.value = override === undefined || override === null ? "" : override;
  field.append(input);
  if (limits) {
    const hint = document.createElement("small");
    hint.className = "settings-hint";
    hint.textContent = `${formatNumber(limits.min)}–${formatNumber(limits.max)}`;
    field.append(hint);
  }
  return field;
}

function settingsField(name) {
  const wrap = document.createElement("div");
  wrap.className = "settings-field";
  const label = document.createElement("label");
  label.htmlFor = `setting-${name}`;
  label.textContent = SETTING_LABELS[name];
  wrap.append(label);
  return wrap;
}

function setSettingsStatus(message, kind) {
  settingsStatus.hidden = false;
  settingsStatus.textContent = message;
  settingsStatus.className = `settings-status ${kind || ""}`;
}

function clearSettingsStatus() {
  settingsStatus.hidden = true;
  settingsStatus.textContent = "";
}

/* ---------- session log (client-side; the server keeps no session listing) ---------- */

function loadSessions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveSessions() {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.slice(0, 50)));
}

function findSession(id) {
  return sessions.find((session) => session.id === id) || null;
}

function generateSessionId() {
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function generateRequestId() {
  return `r-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function recordTurn(sessionId, query, answerId) {
  let session = findSession(sessionId);
  if (!session) {
    session = { id: sessionId, title: query, turns: [], updated: 0 };
    sessions.unshift(session);
  }
  session.turns.push({ query, answer_id: answerId, at: Date.now() });
  session.updated = Date.now();
  sessions = [session, ...sessions.filter((entry) => entry.id !== sessionId)];
  saveSessions();
}

function renderSessionState() {
  const session = activeSessionId ? findSession(activeSessionId) : null;
  if (session) {
    const prefix = activeSessionLive === true ? "Continuing" : "Checking session";
    activeSessionLabel.textContent = `${prefix}: ${truncate(session.title, 48)}`;
    newSessionButton.hidden = false;
  } else {
    activeSessionLabel.textContent = "New session";
    newSessionButton.hidden = true;
  }
}

function renderHistory() {
  historyList.replaceChildren();
  const previous = sessions.filter((session) => session.id !== activeSessionId);
  historySection.hidden = !previous.length;
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
    main.addEventListener("click", () => resumeSession(session));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "history-remove";
    remove.setAttribute("aria-label", "Remove this session from the log");
    remove.textContent = "✕";
    remove.addEventListener("click", () => {
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
  localStorage.setItem(ACTIVE_KEY, session.id);
  renderSessionState();
  renderHistory();
  const lastTurn = session.turns[session.turns.length - 1];
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
    renderAnswer(current, lastTurn.query);
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
    applyDetailsVisibility();
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
  localStorage.removeItem(ACTIVE_KEY);
}

function removeSession(sessionId) {
  sessions = sessions.filter((entry) => entry.id !== sessionId);
  saveSessions();
  if (activeSessionId === sessionId) detachActiveSession();
}

function clearResult() {
  current = null;
  packetLoadedFor = null;
  result.hidden = true;
}

function truncate(text, max) {
  const value = String(text);
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
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

/* ---------- details visibility ---------- */

function applyDetailsVisibility() {
  detailsSection.hidden = !detailsToggle.checked;
  if (detailsToggle.checked && current && packetLoadedFor !== current.evidence_packet_id) {
    loadEvidencePacket();
  }
}

async function loadEvidencePacket() {
  const packetId = current.evidence_packet_id;
  try {
    const packet = await requestJson(`/api/evidence-packets/${packetId}`);
    packetLoadedFor = packetId;
    renderPlan(packet);
    renderEvidencePacket(packet);
  } catch (error) {
    const message = `Could not load the persisted evidence packet: ${error.message}`;
    setPanel("#d-plan", emptyMessage(message));
    setPanel("#d-evidence", emptyMessage(message));
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body?.error?.message || `Request failed (${response.status}).`);
    error.status = response.status;
    error.code = body?.error?.code;
    throw error;
  }
  return body;
}

function startRequestFeedback(request) {
  cancelRequestButton.hidden = false;
  cancelRequestButton.disabled = false;
  renderRequestFeedback(request);
  request.elapsedTimer = window.setInterval(() => renderRequestFeedback(request), 1000);
  request.progressTimer = window.setInterval(async () => {
    try {
      const progress = await requestJson(`/api/asks/${request.requestId}/progress`);
      if (activeRequest === request && !request.cancelled) {
        request.progress = progress;
        renderRequestFeedback(request);
      }
    } catch {
      // Preserve the generic status when a server cannot provide telemetry.
    }
  }, 1000);
}

function stopRequestFeedback(request) {
  window.clearInterval(request.elapsedTimer);
  window.clearInterval(request.progressTimer);
  if (activeRequest === request) cancelRequestButton.hidden = true;
}

function renderRequestFeedback(request) {
  if (activeRequest !== request || request.cancelled) return;
  const elapsed = request.progress?.elapsed_seconds ?? (Date.now() - request.startedAt) / 1000;
  const phaseLabels = {
    planning: "Planning the search",
    keyword_search: "Running keyword search",
    semantic_search: "Running semantic search",
    answer_generation: "Generating the answer",
  };
  const message = phaseLabels[request.progress?.phase] || "Searching your indexed materials…";
  setStatus(`${message} (${formatElapsed(elapsed)})`, "working");
}

function formatElapsed(seconds) {
  return `${Math.max(0, Math.floor(Number(seconds) || 0))}s`;
}

/* ---------- simple view ---------- */

function renderAnswer(payload, queryText) {
  result.hidden = false;
  const askedQuery = document.querySelector("#asked-query");
  askedQuery.textContent = queryText || "";

  const answerCard = document.querySelector("#answer-card");
  const answerState = document.querySelector("#answer-state");
  const answerStatus = payload.answer_status || "answered";
  const isFailure = answerStatus !== "answered";
  answerCard.classList.toggle("answer-failure", isFailure);
  answerState.hidden = !isFailure;
  answerState.textContent = answerStatus === "validation_failed"
    ? "Answer generation failed validation"
    : "Insufficient source evidence";

  const answerRoot = document.querySelector("#answer-text");
  answerRoot.replaceChildren();
  const paragraphs = String(payload.answer_body).split(/\n+/).filter((p) => p.trim());
  (paragraphs.length ? paragraphs : ["No answer text was returned."]).forEach((text) => {
    const p = document.createElement("p");
    p.textContent = text.trim();
    p.dir = "auto";
    answerRoot.append(p);
  });

  renderSources(payload.references || []);
  renderCitations(payload.citations || []);
  renderLimitations(payload.limitations || []);
  renderCoverage(payload.coverage || {});
  renderTrace(payload);
  setPanel("#d-plan", emptyMessage("Loading persisted evidence packet…"));
  setPanel("#d-evidence", emptyMessage("Loading persisted evidence packet…"));
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSources(references) {
  const root = document.querySelector("#sources");
  root.replaceChildren();
  if (!references.length) return;
  const label = document.createElement("span");
  label.className = "sources-label";
  label.textContent = "Sources";
  root.append(label);
  references.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.dir = "auto";
    chip.title = `${item.course} · ${item.file_path} · ${item.location_label}`;
    const id = document.createElement("b");
    id.textContent = item.citation_id;
    chip.append(id, ` ${fileName(item.file_path)} · ${item.location_label}`);
    root.append(chip);
  });
}

/* ---------- detail panels ---------- */

function renderCitations(citations) {
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
  if (unshownWeaknesses.length) projected.additional_weaknesses = unshownWeaknesses;
  setPanel("#d-coverage", definitionGrid(projected));
}

function renderPlan(packet) {
  const fragment = document.createDocumentFragment();
  fragment.append(
    subheading("Interpreted intent"),
    paragraph(packet.interpreted_intent || "Not recorded."),
    subheading("Query plan"),
    definitionGrid(packet.query_plan || {}),
    subheading("Retrieval settings"),
    definitionGrid(packet.retrieval_settings || {}),
    subheading("Searched"),
    definitionGrid(packet.searched || {}),
  );
  setPanel("#d-plan", fragment);
}

function renderEvidencePacket(packet) {
  const fragment = document.createDocumentFragment();
  const items = packet.evidence || [];

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
    title.textContent = `E${index + 1} · ${item.course}`;
    head.append(title, badge(item.retrieval_method), badge(`rank ${item.rank}`), badge(`score ${formatNumber(item.score)}`), badge(`${item.token_count} tokens`));
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
  fragment.append(definitionGrid({
    answer_id: payload.answer_id,
    search_run_id: payload.search_run_id,
    evidence_packet_id: payload.evidence_packet_id,
  }));
  const links = document.createElement("div");
  links.className = "api-links";
  [
    ["Answer JSON", `/api/answers/${payload.answer_id}`],
    ["Coverage JSON", `/api/search-runs/${payload.search_run_id}/coverage`],
    ["Evidence packet JSON", `/api/evidence-packets/${payload.evidence_packet_id}`],
  ].forEach(([label, href]) => {
    const a = document.createElement("a");
    a.href = href;
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = label;
    links.append(a);
  });
  fragment.append(links);
  setPanel("#d-trace", fragment);
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
  root.replaceChildren(node);
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

function fileName(path) {
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function setBusy(busy, message = "") {
  askButton.disabled = busy;
  askButton.classList.toggle("busy", busy);
  askButton.querySelector(".button-label").textContent = busy ? "Working…" : "Ask";
  if (!busy) {
    cancelRequestButton.hidden = true;
    cancelRequestButton.disabled = false;
  }
  if (message) setStatus(message, "working");
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
