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

const SESSIONS_KEY = "uni-rag-sessions";
const ACTIVE_KEY = "uni-rag-active-session";
const DETAILS_KEY = "uni-rag-details";

let current = null;
let packetLoadedFor = null;
let sessions = loadSessions();
let activeSessionId = localStorage.getItem(ACTIVE_KEY);
if (activeSessionId && !findSession(activeSessionId)) activeSessionId = null;

detailsToggle.checked = localStorage.getItem(DETAILS_KEY) === "1";
applyDetailsVisibility();
renderSessionState();
renderHistory();

detailsToggle.addEventListener("change", () => {
  localStorage.setItem(DETAILS_KEY, detailsToggle.checked ? "1" : "0");
  applyDetailsVisibility();
});

newSessionButton.addEventListener("click", () => {
  activeSessionId = null;
  current = null;
  packetLoadedFor = null;
  localStorage.removeItem(ACTIVE_KEY);
  result.hidden = true;
  clearStatus();
  queryInput.value = "";
  renderSessionState();
  renderHistory();
  queryInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  setBusy(true, "Searching your indexed materials…");
  const sessionId = activeSessionId || generateSessionId();
  try {
    current = await requestJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: sessionId }),
    });
    packetLoadedFor = null;
    activeSessionId = sessionId;
    localStorage.setItem(ACTIVE_KEY, sessionId);
    recordTurn(sessionId, query, current.answer_id);
    renderSessionState();
    renderHistory();
    renderAnswer(current, query);
    clearStatus();
    applyDetailsVisibility();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});

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
    activeSessionLabel.textContent = `Continuing: ${truncate(session.title, 48)}`;
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
  activeSessionId = session.id;
  localStorage.setItem(ACTIVE_KEY, session.id);
  renderSessionState();
  renderHistory();
  const lastTurn = session.turns[session.turns.length - 1];
  if (!lastTurn?.answer_id) {
    setStatus("Session resumed. Ask a follow-up question.", "working");
    queryInput.focus();
    return;
  }
  setBusy(true, "Loading the session's latest answer…");
  try {
    current = await requestJson(`/api/answers/${lastTurn.answer_id}`);
    packetLoadedFor = null;
    renderAnswer(current, lastTurn.query);
    clearStatus();
    applyDetailsVisibility();
  } catch (error) {
    setStatus(`Session resumed, but its stored answer could not be loaded: ${error.message}`, "error");
  } finally {
    setBusy(false);
    queryInput.focus();
  }
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
    throw new Error(body?.error?.message || `Request failed (${response.status}).`);
  }
  return body;
}

/* ---------- simple view ---------- */

function renderAnswer(payload, queryText) {
  result.hidden = false;
  document.querySelector("#asked-query").textContent = queryText || "";

  const answerRoot = document.querySelector("#answer-text");
  answerRoot.replaceChildren();
  const paragraphs = String(payload.answer_text || "").split(/\n{2,}/).filter((p) => p.trim());
  (paragraphs.length ? paragraphs : ["No answer text was returned."]).forEach((text) => {
    const p = document.createElement("p");
    p.textContent = text.trim();
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
  if (!limitations.length) {
    setPanel("#d-limitations", emptyMessage("No limitations reported."));
    return;
  }
  setPanel("#d-limitations", stringList(limitations));
}

function renderCoverage(coverage) {
  setPanel("#d-coverage", definitionGrid(coverage));
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

  if (packet.weaknesses?.length) {
    fragment.append(subheading("Weaknesses"), stringList(packet.weaknesses));
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
