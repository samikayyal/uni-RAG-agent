(function attachBrowserState(root, factory) {
  const api = factory();
  root.UniRagBrowserState = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window, function buildBrowserState() {
  "use strict";

  function selectStore(mode, localStore, tabStore) {
    return mode === "public" ? tabStore : localStore;
  }

  function loadSessions(store, key) {
    try {
      const parsed = JSON.parse(store.getItem(key) || "[]");
      if (Array.isArray(parsed)) return parsed;
    } catch {
      // Fall through to remove the invalid value below.
    }
    store.removeItem(key);
    return [];
  }

  function saveSessions(store, key, values, limit, pruneOnQuota) {
    const retained = values.slice(0, limit);
    while (retained.length) {
      try {
        store.setItem(key, JSON.stringify(retained));
        return retained;
      } catch (error) {
        if (!pruneOnQuota || error?.name !== "QuotaExceededError") throw error;
        retained.pop();
      }
    }
    store.setItem(key, "[]");
    return retained;
  }

  function clearSessions(store, sessionsKey, activeKey) {
    store.removeItem(sessionsKey);
    store.removeItem(activeKey);
    return [];
  }

  function tokenState(store, tokenKey, expiryKey, nowSeconds, leewaySeconds = 15) {
    const token = store.getItem(tokenKey);
    const expiresAt = Number(store.getItem(expiryKey) || 0);
    const valid = Boolean(token) && expiresAt > nowSeconds + leewaySeconds;
    return {
      token: valid ? token : null,
      valid,
      renewalRequired: Boolean(token) && !valid,
    };
  }

  function publicResumeDecision(session, { tokenRenewed, serverLive }) {
    const payload = session?.latest?.payload || null;
    if (!payload) return { remove: true, payload: null, continueSession: false };
    return {
      remove: false,
      payload,
      continueSession: !tokenRenewed && serverLive === true,
    };
  }

  function localEmbeddingPreparationModel(payload, changes) {
    // Preparation is a consequence of an explicit profile selection, not of
    // saving other fields while the selector remains on the server default.
    const model = changes?.embedding_model;
    if (typeof model !== "string" || !model.trim()) return null;
    const profile = (payload?.embedding_model_profiles || []).find(
      (candidate) => candidate.model_name === model,
    );
    return profile?.provider === "huggingface" ? profile.model_name : null;
  }

  function randomId(cryptoProvider) {
    if (typeof cryptoProvider?.randomUUID === "function") {
      return cryptoProvider.randomUUID();
    }
    if (typeof cryptoProvider?.getRandomValues !== "function") {
      throw new Error("This browser cannot generate secure request identifiers.");
    }
    const bytes = cryptoProvider.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
    return [
      hex.slice(0, 4).join(""),
      hex.slice(4, 6).join(""),
      hex.slice(6, 8).join(""),
      hex.slice(8, 10).join(""),
      hex.slice(10, 16).join(""),
    ].join("-");
  }

  return {
    clearSessions,
    loadSessions,
    localEmbeddingPreparationModel,
    publicResumeDecision,
    randomId,
    saveSessions,
    selectStore,
    tokenState,
  };
});
