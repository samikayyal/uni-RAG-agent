"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const state = require("../src/uni_rag_agent/app/static/browser_state.js");

class MemoryStorage {
  constructor(maxLength = Infinity) {
    this.values = new Map();
    this.maxLength = maxLength;
  }

  getItem(key) {
    return this.values.has(key) ? this.values.get(key) : null;
  }

  setItem(key, value) {
    if (String(value).length > this.maxLength) {
      const error = new Error("storage full");
      error.name = "QuotaExceededError";
      throw error;
    }
    this.values.set(key, String(value));
  }

  removeItem(key) {
    this.values.delete(key);
  }
}

const sessionsKey = "sessions";

test("public tabs isolate settings and restored answer payloads", () => {
  const local = new MemoryStorage();
  const tabA = new MemoryStorage();
  const tabB = new MemoryStorage();
  const storeA = state.selectStore("public", local, tabA);
  const storeB = state.selectStore("public", local, tabB);
  storeA.setItem("settings", JSON.stringify({ final_top_k: 3 }));
  storeB.setItem("settings", JSON.stringify({ final_top_k: 9 }));
  const payload = { answer_body: "restored", evidence_packet: { evidence: [] } };
  state.saveSessions(storeA, sessionsKey, [{
    id: "a",
    latest: { query: "q", payload },
    turns: [],
  }], 20, true);

  assert.deepEqual(JSON.parse(storeA.getItem("settings")), { final_top_k: 3 });
  assert.deepEqual(JSON.parse(storeB.getItem("settings")), { final_top_k: 9 });
  assert.equal(state.loadSessions(storeB, sessionsKey).length, 0);
  assert.deepEqual(state.loadSessions(storeA, sessionsKey)[0].latest.payload, payload);
});

test("public history keeps newest twenty and prunes oldest on quota pressure", () => {
  const sessions = Array.from({ length: 25 }, (_, index) => ({
    id: `session-${index}`,
    title: "x".repeat(20),
  }));
  const normal = new MemoryStorage();
  const retained = state.saveSessions(normal, sessionsKey, sessions, 20, true);
  assert.equal(retained.length, 20);
  assert.equal(retained[0].id, "session-0");
  assert.equal(retained.at(-1).id, "session-19");

  const constrained = new MemoryStorage(170);
  const pruned = state.saveSessions(constrained, sessionsKey, sessions, 20, true);
  assert.ok(pruned.length > 0 && pruned.length < 20);
  assert.equal(pruned[0].id, "session-0");
  assert.equal(state.loadSessions(constrained, sessionsKey).length, pruned.length);
});

test("invalid stored session history is removed after a safe empty fallback", () => {
  const store = new MemoryStorage();
  store.setItem(sessionsKey, "not json");

  assert.deepEqual(state.loadSessions(store, sessionsKey), []);
  assert.equal(store.getItem(sessionsKey), null);

  store.setItem(sessionsKey, JSON.stringify({ not: "a session list" }));
  assert.deepEqual(state.loadSessions(store, sessionsKey), []);
  assert.equal(store.getItem(sessionsKey), null);
});

test("clearing history removes stored sessions and the active session", () => {
  const store = new MemoryStorage();
  store.setItem(sessionsKey, JSON.stringify([{ id: "session-one" }]));
  store.setItem("active-session", "session-one");
  store.setItem("theme", "dark");

  assert.deepEqual(
    state.clearSessions(store, sessionsKey, "active-session"),
    [],
  );
  assert.equal(store.getItem(sessionsKey), null);
  assert.equal(store.getItem("active-session"), null);
  assert.equal(store.getItem("theme"), "dark");
});

test("expired token detaches server context without deleting restored answer", () => {
  const store = new MemoryStorage();
  store.setItem("token", "signed-token");
  store.setItem("expiry", "100");
  const expired = state.tokenState(store, "token", "expiry", 100);
  assert.equal(expired.valid, false);
  assert.equal(expired.renewalRequired, true);

  const session = {
    latest: { payload: { answer_body: "still visible" } },
  };
  const renewed = state.publicResumeDecision(session, {
    tokenRenewed: true,
    serverLive: true,
  });
  assert.equal(renewed.continueSession, false);
  assert.equal(renewed.payload.answer_body, "still visible");
  assert.equal(renewed.remove, false);

  const live = state.publicResumeDecision(session, {
    tokenRenewed: false,
    serverLive: true,
  });
  assert.equal(live.continueSession, true);
  assert.equal(state.publicResumeDecision({}, {
    tokenRenewed: false,
    serverLive: true,
  }).remove, true);
});

test("only a selected local profile is prepared before saving settings", () => {
  const payload = {
    defaults: { embedding_model: "Qwen/Qwen3-Embedding-8B" },
    embedding_model_profiles: [
      { model_name: "google/embeddinggemma-300m", provider: "huggingface" },
      { model_name: "Qwen/Qwen3-Embedding-8B", provider: "nebius" },
    ],
  };

  assert.equal(
    state.localEmbeddingPreparationModel(payload, {
      embedding_model: "google/embeddinggemma-300m",
    }),
    "google/embeddinggemma-300m",
  );
  assert.equal(
    state.localEmbeddingPreparationModel(payload, {
      embedding_model: "Qwen/Qwen3-Embedding-8B",
    }),
    null,
  );
  assert.equal(state.localEmbeddingPreparationModel(payload, {}), null);
  assert.equal(
    state.localEmbeddingPreparationModel(
      {
        ...payload,
        defaults: { embedding_model: "google/embeddinggemma-300m" },
      },
      { embedding_model: null, final_top_k: 4 },
    ),
    null,
  );
  assert.equal(
    state.localEmbeddingPreparationModel(
      {
        ...payload,
        defaults: { embedding_model: "google/embeddinggemma-300m" },
      },
      {},
    ),
    null,
  );
});
