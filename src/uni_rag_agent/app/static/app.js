const form = document.querySelector("#ask-form");
const queryInput = document.querySelector("#query");
const sessionInput = document.querySelector("#session-id");
const askButton = document.querySelector("#ask-button");
const statusBox = document.querySelector("#status");
const result = document.querySelector("#result");
let current = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true, "Searching indexed materials and building an evidence-grounded answer…");
  const payload = { query: queryInput.value };
  if (sessionInput.value.trim()) payload.session_id = sessionInput.value.trim();
  try {
    current = await requestJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderAnswer(current);
    setStatus("Answer complete.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

document.querySelector("#refresh-coverage").addEventListener("click", async () => {
  if (!current) return;
  try {
    const coverage = await requestJson(`/api/search-runs/${current.search_run_id}/coverage`);
    current.coverage = coverage;
    renderCoverage(coverage);
    setStatus("Persisted coverage reloaded.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

document.querySelector("#load-evidence").addEventListener("click", async () => {
  if (!current) return;
  try {
    const packet = await requestJson(`/api/evidence-packets/${current.evidence_packet_id}`);
    renderEvidence(packet.evidence || []);
    setStatus("Persisted evidence loaded.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body?.error?.message || `Request failed (${response.status}).`);
  }
  return body;
}

function renderAnswer(payload) {
  result.classList.remove("hidden");
  document.querySelector("#answer-text").textContent = payload.answer_text;
  document.querySelector("#trace-ids").textContent =
    `answer ${payload.answer_id} · run ${payload.search_run_id} · packet ${payload.evidence_packet_id}`;
  renderReferences(payload.references || []);
  renderList(document.querySelector("#limitations"), payload.limitations || [], "No limitations reported.");
  renderCoverage(payload.coverage || {});
  renderEvidence([]);
  setLink("#answer-api-link", `/api/answers/${payload.answer_id}`);
  setLink("#coverage-api-link", `/api/search-runs/${payload.search_run_id}/coverage`);
  setLink("#evidence-api-link", `/api/evidence-packets/${payload.evidence_packet_id}`);
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderReferences(items) {
  const root = document.querySelector("#references");
  root.replaceChildren();
  if (!items.length) {
    root.append(emptyMessage("No citations were needed for this answer."));
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "reference";
    const title = document.createElement("strong");
    title.textContent = `${item.citation_id} · ${item.course}`;
    const path = document.createElement("span");
    path.textContent = item.file_path;
    const location = document.createElement("small");
    location.textContent = `${item.source_type} · ${item.location_label}`;
    card.append(title, path, location);
    root.append(card);
  });
}

function renderCoverage(coverage) {
  const root = document.querySelector("#coverage");
  root.replaceChildren();
  const sections = [
    ["Status", [coverage.status || "unknown"]],
    ["Searched courses", coverage.searched_courses || []],
    ["Searched indexes", coverage.searched_indexes || []],
    ["Found in courses", coverage.courses_with_chunk_hits || []],
    ["Found in indexes", coverage.indexes_with_chunk_hits || []],
    ["Missing course coverage", coverage.courses_without_chunk_hits || []],
    ["Missing index coverage", coverage.indexes_without_chunk_hits || []],
    ["Weaknesses", coverage.weaknesses || []],
  ];
  sections.forEach(([label, values]) => {
    const block = document.createElement("div");
    const heading = document.createElement("strong");
    heading.textContent = label;
    const value = document.createElement("p");
    value.textContent = values.length ? values.join(" · ") : "None";
    block.append(heading, value);
    root.append(block);
  });
}

function renderEvidence(items) {
  const root = document.querySelector("#evidence");
  root.replaceChildren();
  if (!items.length) {
    root.append(emptyMessage("Select “Load evidence” to inspect the persisted packet."));
    return;
  }
  items.forEach((item, index) => {
    const article = document.createElement("article");
    article.className = "evidence-item";
    const title = document.createElement("strong");
    title.textContent = `E${index + 1} · ${item.course} · ${item.location?.label || "location unavailable"}`;
    const path = document.createElement("small");
    path.textContent = item.file;
    const text = document.createElement("p");
    text.textContent = item.text;
    article.append(title, path, text);
    root.append(article);
  });
}

function renderList(root, items, fallback) {
  root.replaceChildren();
  (items.length ? items : [fallback]).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    root.append(li);
  });
}

function emptyMessage(text) {
  const node = document.createElement("p");
  node.className = "empty";
  node.textContent = text;
  return node;
}

function setLink(selector, href) {
  const link = document.querySelector(selector);
  link.href = href;
}

function setBusy(busy, message = "") {
  askButton.disabled = busy;
  askButton.textContent = busy ? "Working…" : "Ask archive";
  if (message) setStatus(message, "working");
}

function setStatus(message, kind) {
  statusBox.textContent = message;
  statusBox.className = `status ${kind || ""}`;
}
