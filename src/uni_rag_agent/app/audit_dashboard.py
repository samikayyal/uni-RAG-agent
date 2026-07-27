"""Localhost-only SQLite-backed Firestore ask-audit dashboard."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import Config, load_config
from .ask_audit import AskAuditError, FirestoreAskAuditStore
from .audit_cache import AuditCache, AuditCacheError, GeoIpResolver


def create_audit_dashboard(
    *,
    config_loader: Any = load_config,
    store: Any = None,
    geoip_database: Path | None = None,
    cache_database: Path = Path("data/ask_audit_cache.sqlite"),
    offline: bool = False,
    rebuild_cache: bool = False,
    refresh_locations: bool = False,
) -> FastAPI:
    """Create the local viewer after synchronizing its generated SQLite cache."""

    if offline and rebuild_cache:
        raise ValueError("--rebuild-cache cannot be used with --offline.")
    cache = AuditCache(cache_database)
    cache.initialize()
    geo = GeoIpResolver(geoip_database)
    config: Config | None = None
    audit_store = None
    if not offline:
        config = config_loader()
        if not config.firestore_project_id:
            raise RuntimeError(
                "UNI_RAG_FIRESTORE_PROJECT_ID is required for the ask dashboard."
            )
        audit_store = store or FirestoreAskAuditStore(config)
        try:
            cache.sync(audit_store, geo, rebuild=rebuild_cache)
        except Exception as exc:
            if cache.count() == 0:
                raise RuntimeError(
                    "Firestore synchronization failed and the audit cache is empty. "
                    "Verify credentials/connectivity, or use --offline with an existing cache."
                ) from exc
    if refresh_locations:
        cache.refresh_locations(geo)
    app = FastAPI(
        title="Uni RAG Ask Audit",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def local_security_headers(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "object-src 'none'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; img-src 'self' data:;"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(AskAuditError)
    @app.exception_handler(AuditCacheError)
    async def handle_audit_error(_: Any, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "ask_audit_unavailable",
                    "message": str(exc),
                }
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    sync_lock = asyncio.Lock()
    sync_in_progress = False

    def _metadata() -> dict[str, object]:
        state = cache.state()
        return {
            "project": config.firestore_project_id if config is not None else None,
            "database": config.firestore_database if config is not None else None,
            "cache_path": str(cache.path),
            "offline": offline,
            "cached_count": cache.count(),
            "last_successful_sync": state["last_success_at"],
            "last_error": state["last_error"],
            "sync_state": (
                "offline"
                if offline
                else "synchronizing"
                if sync_in_progress
                else "stale"
                if state["last_error"]
                else "current"
            ),
            "geoip": {
                "ready": geo.ready,
                "database_path": (
                    str(geo.database_path) if geo.database_path is not None else None
                ),
                "build_epoch": geo.build_epoch,
                "message": geo.error,
            },
        }

    @app.get("/api/meta")
    async def metadata() -> dict[str, object]:
        return _metadata()

    @app.get("/api/asks")
    async def list_asks(
        limit: int = Query(default=100, ge=1, le=500),
        before: str | None = None,
    ) -> dict[str, object]:
        try:
            before_value = _parse_cursor(before)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        records, next_before = cache.list_recent(limit=limit, before=before_value)
        return {
            "records": records,
            "next_before": next_before,
        }

    @app.get("/api/asks/{audit_id}", response_model=None)
    async def get_ask(audit_id: str) -> Any:
        if len(audit_id) != 32 or any(
            character not in "0123456789abcdef" for character in audit_id
        ):
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": "Ask not found."}},
            )
        record = cache.get(audit_id)
        if record is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": "Ask not found."}},
            )
        return record

    @app.post("/api/sync")
    async def synchronize() -> dict[str, object]:
        nonlocal sync_in_progress
        if offline:
            return _metadata()
        assert audit_store is not None
        async with sync_lock:
            sync_in_progress = True
            try:
                await asyncio.to_thread(cache.sync, audit_store, geo)
            except Exception as exc:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "code": "ask_audit_unavailable",
                            "message": "Firestore synchronization failed; cached data may be stale.",
                        }
                    },
                )
            finally:
                sync_in_progress = False
        return _metadata()

    return app


def _parse_cursor(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("before must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("before must include a timezone")
    return parsed.astimezone(UTC)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Uni RAG Ask Audit</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07100d;
      --panel: #101a16;
      --panel-2: #15231d;
      --line: #294036;
      --text: #edf7f1;
      --muted: #9eb5a9;
      --accent: #7ee2aa;
      --amber: #f6c86d;
      --red: #ff8c8c;
      --blue: #83bfff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 82% -10%, #16392b 0, transparent 33rem),
        var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    button, input, select { font: inherit; }
    .shell { max-width: 1500px; margin: auto; padding: 32px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: end; }
    .eyebrow { color: var(--accent); letter-spacing: .13em; text-transform: uppercase; font-size: 11px; font-weight: 800; }
    h1 { margin: 5px 0 0; font-size: clamp(28px, 4vw, 48px); letter-spacing: -.04em; }
    .meta { color: var(--muted); text-align: right; }
    .notice { display: none; margin-top: 20px; border: 1px solid #6d5727; background: #2d2615; color: #ffe3a3; padding: 12px 14px; border-radius: 12px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 26px 0 18px; }
    .card { background: color-mix(in srgb, var(--panel) 94%, transparent); border: 1px solid var(--line); border-radius: 16px; padding: 18px; }
    .card b { display: block; font-size: 28px; letter-spacing: -.03em; margin-top: 8px; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .toolbar { display: grid; grid-template-columns: 1fr 220px auto; gap: 12px; margin-bottom: 14px; }
    input, select, button {
      border: 1px solid var(--line); background: var(--panel-2); color: var(--text);
      border-radius: 11px; padding: 11px 13px;
    }
    button { cursor: pointer; font-weight: 700; }
    button:hover { border-color: var(--accent); }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 16px; background: var(--panel); }
    table { width: 100%; border-collapse: collapse; min-width: 1050px; }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; text-align: left; padding: 13px 14px; border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--panel); }
    td { padding: 13px 14px; border-bottom: 1px solid #1d3028; vertical-align: top; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: #17271f; }
    .query { max-width: 520px; font-weight: 650; }
    .sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .badge { display: inline-flex; border-radius: 999px; padding: 4px 9px; font-size: 11px; font-weight: 800; background: #24382f; color: var(--accent); }
    .badge.failed, .badge.rejected_validation, .badge.rejected_quota, .badge.rejected_busy, .badge.rejected_replay, .badge.rejected_in_progress, .badge.rejected_session_owner, .badge.failed_infrastructure { color: var(--red); background: #3b2020; }
    .badge.cancelled, .badge.timed_out { color: var(--amber); background: #3a301c; }
    .load { width: 100%; margin-top: 14px; }
    dialog { width: min(1000px, calc(100vw - 30px)); max-height: 88vh; color: var(--text); background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 0; }
    dialog::backdrop { background: #000b; backdrop-filter: blur(5px); }
    .dialog-head { display: flex; justify-content: space-between; align-items: center; padding: 17px 20px; border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--panel); }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; padding: 20px; color: #cfe6d9; font: 12px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .empty { text-align: center; padding: 50px; color: var(--muted); }
    @media (max-width: 800px) {
      .shell { padding: 20px 14px; }
      header { align-items: start; flex-direction: column; }
      .meta { text-align: left; }
      .cards { grid-template-columns: repeat(2, 1fr); }
      .toolbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div><div class="eyebrow">Local SQLite audit cache</div><h1>Ask audit</h1></div>
      <div class="meta" id="meta">Connecting…</div>
    </header>
    <div class="notice" id="notice"></div>
    <section class="cards">
      <div class="card"><span class="label">Loaded asks</span><b id="total">—</b></div>
      <div class="card"><span class="label">Completed</span><b id="completed">—</b></div>
      <div class="card"><span class="label">Rejected / failed</span><b id="failed">—</b></div>
      <div class="card"><span class="label">Countries</span><b id="countries">—</b></div>
    </section>
    <section class="toolbar">
      <input id="search" type="search" placeholder="Search questions, IPs, locations, request IDs…">
      <select id="status"><option value="">All statuses</option></select>
      <button id="refresh">Refresh</button>
    </section>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Received</th><th>Status</th><th>Question</th><th>Location</th><th>IP address</th><th>Duration</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="empty" id="empty">Loading audit records…</div>
    </div>
    <button class="load" id="more" hidden>Load older asks</button>
  </main>
  <dialog id="detail">
    <div class="dialog-head"><strong>Complete Firestore record</strong><button id="close">Close</button></div>
    <pre id="json"></pre>
  </dialog>
  <script>
    const state = { records: [], next: null };
    const $ = id => document.getElementById(id);
    const esc = value => String(value ?? "");
    const time = value => value ? new Date(value).toLocaleString() : "—";
    const duration = value => Number.isFinite(value) ? `${(value / 1000).toFixed(2)}s` : "—";
    const locationText = record => {
      const g = record.location;
      if (!g) return "Unresolved";
      return [g.city, g.region, g.country].filter(Boolean).join(", ") || "Unresolved";
    };
    async function request(path, options = {}) {
      const response = await fetch(path, { cache: "no-store", ...options });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.error?.message || "Request failed");
      return data;
    }
    async function load(reset = false) {
      if (reset) { state.records = []; state.next = null; }
      $("empty").hidden = false;
      $("empty").textContent = "Loading audit records…";
      const suffix = state.next ? `&before=${encodeURIComponent(state.next)}` : "";
      try {
        const data = await request(`/api/asks?limit=100${suffix}`);
        state.records.push(...data.records);
        state.next = data.next_before;
        $("more").hidden = !state.next;
        refreshStatuses();
        render();
      } catch (error) {
        $("empty").hidden = false;
        $("empty").textContent = error.message;
      }
    }
    function refreshStatuses() {
      const selected = $("status").value;
      const values = [...new Set(state.records.map(r => r.status).filter(Boolean))].sort();
      $("status").replaceChildren(new Option("All statuses", ""), ...values.map(v => new Option(v, v)));
      $("status").value = values.includes(selected) ? selected : "";
    }
    function filtered() {
      const needle = $("search").value.trim().toLowerCase();
      const status = $("status").value;
      return state.records.filter(record => {
        if (status && record.status !== status) return false;
        if (!needle) return true;
        return JSON.stringify({
          q: record.query, ip: record.client?.ip_address, loc: record.location,
          request: record.request_id, session: record.session_id, status: record.status
        }).toLowerCase().includes(needle);
      });
    }
    function render() {
      const records = filtered();
      $("total").textContent = state.records.length.toLocaleString();
      $("completed").textContent = state.records.filter(r => r.status === "completed").length.toLocaleString();
      $("failed").textContent = state.records.filter(r => r.status !== "completed" && r.status !== "accepted" && r.status !== "received").length.toLocaleString();
      $("countries").textContent = new Set(state.records.map(r => r.location?.country_code).filter(Boolean)).size.toLocaleString();
      $("rows").replaceChildren(...records.map(record => {
        const tr = document.createElement("tr");
        const cells = [
          [time(record.received_at), record.request_id || record.audit_id],
          [record.status || "unknown", record.outcome || record.error?.code || ""],
          [record.query || "—", record.models?.embedding_model || record.requested_retrieval_settings?.embedding_model || ""],
          [locationText(record), record.location?.time_zone || ""],
          [record.client?.ip_address || "—", record.client?.accept_language || ""],
          [duration(record.duration_ms), record.timing?.phases?.map(p => p.phase).join(" → ") || ""]
        ];
        cells.forEach(([primary, secondary], index) => {
          const td = document.createElement("td");
          if (index === 1) {
            const badge = document.createElement("span");
            badge.className = `badge ${record.status || ""}`;
            badge.textContent = primary;
            td.append(badge);
          } else {
            const main = document.createElement("div");
            if (index === 2) main.className = "query";
            main.textContent = primary;
            td.append(main);
          }
          if (secondary) {
            const sub = document.createElement("div");
            sub.className = "sub";
            sub.textContent = secondary;
            td.append(sub);
          }
          tr.append(td);
        });
        tr.addEventListener("click", () => show(record.audit_id));
        return tr;
      }));
      $("empty").hidden = records.length > 0;
      if (!records.length) $("empty").textContent = "No matching asks.";
    }
    async function show(id) {
      $("json").textContent = "Loading…";
      $("detail").showModal();
      try { $("json").textContent = JSON.stringify(await request(`/api/asks/${id}`), null, 2); }
      catch (error) { $("json").textContent = error.message; }
    }
    async function boot() {
      try {
        const meta = await request("/api/meta");
        $("notice").style.display = "none";
        const source = meta.offline ? "Offline cache" : `${meta.project} · ${meta.database}`;
        $("meta").textContent = `${source} · ${meta.cached_count.toLocaleString()} cached`;
        if (meta.sync_state === "synchronizing") {
          $("notice").textContent = "Synchronizing Firestore into the local SQLite cache…";
          $("notice").style.display = "block";
        } else if (meta.sync_state === "stale") {
          $("notice").textContent = `Stale cached data. Last successful sync: ${time(meta.last_successful_sync)}. ${meta.last_error || ""}`;
          $("notice").style.display = "block";
        } else if (meta.offline) {
          $("notice").textContent = "Offline mode: Refresh reloads only the local SQLite cache.";
          $("notice").style.display = "block";
        } else if (!meta.geoip.ready) {
          $("notice").textContent = meta.geoip.message;
          $("notice").style.display = "block";
        }
      } catch (error) { $("meta").textContent = error.message; }
      await load(true);
    }
    $("search").addEventListener("input", render);
    $("status").addEventListener("change", render);
    $("refresh").addEventListener("click", async () => {
      $("notice").textContent = "Synchronizing Firestore into the local SQLite cache…";
      $("notice").style.display = "block";
      try { await request("/api/sync", { method: "POST" }); } catch (_) { /* metadata exposes stale state */ }
      await boot();
    });
    $("more").addEventListener("click", () => load(false));
    $("close").addEventListener("click", () => $("detail").close());
    boot();
  </script>
</body>
</html>
"""


__all__ = ["GeoIpResolver", "create_audit_dashboard"]
