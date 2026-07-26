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

  return {
    loadSessions,
    localEmbeddingPreparationModel,
    publicResumeDecision,
    saveSessions,
    selectStore,
    tokenState,
  };
});
