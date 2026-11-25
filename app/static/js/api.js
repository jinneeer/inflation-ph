/* API client with abortable GETs, in-memory dedupe and a small debounce helper. */

const cache = new Map();
const pending = new Map();           // url -> in-flight promise
export const inflight = new Map();
export const lastMeta = new Map();   // panel_id -> {row_count, elapsed_ms, rtt_ms, cache, params}
let seq = 0;

function record(meta, t0, cacheState = "miss") {
  if (!meta?.panel_id || meta.row_count === undefined) return;
  lastMeta.set(meta.panel_id, {
    ...meta,
    cache: cacheState === "hit" ? "hit" : meta.cache || cacheState,
    rtt_ms: Math.round(performance.now() - t0),
    at: Date.now(),
  });
  window.dispatchEvent(new CustomEvent("api:meta", { detail: meta.panel_id }));
}

function qs(params = {}) {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") p.set(k, v);
  });
  const s = p.toString();
  return s ? `?${s}` : "";
}

// fetch() rejects with a bare TypeError ("Failed to fetch") when the server is unreachable,
// e.g. while `uvicorn --reload` restarts. Retry those briefly; HTTP errors and aborts are not retried.
const RETRY_DELAYS = [400, 1200, 3000];
async function fetchWithRetry(url, opts = {}) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fetch(url, opts);
    } catch (err) {
      if (err.name === "AbortError" || opts.signal?.aborted) throw err;
      if (attempt >= RETRY_DELAYS.length) {
        throw new Error("Can't reach the dashboard server. Check that it is still running, then retry.");
      }
      await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
    }
  }
}

export async function get(path, params = {}, { fresh = false } = {}) {
  const url = `/api${path}${qs(params)}`;
  const t0 = performance.now();
  if (!fresh && cache.has(url)) {
    const cached = cache.get(url);
    record(cached?.meta, t0, "hit");
    return cached;
  }
  // Concurrent callers for the same URL share one request; failures are not cached.
  if (!fresh && pending.has(url)) return pending.get(url);
  const ctrl = new AbortController();
  inflight.set(url, ctrl);
  const p = (async () => {
    try {
      const res = await fetchWithRetry(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error(`${path} → ${res.status}`);
      const body = await res.json();
      cache.set(url, body);
      record(body?.meta, t0);
      return body;
    } finally {
      inflight.delete(url);
      pending.delete(url);
    }
  })();
  pending.set(url, p);
  return p;
}

export async function post(path, payload) {
  const t0 = performance.now();
  const res = await fetchWithRetry(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  const body = await res.json();
  record(body?.meta, t0);
  return body;
}

export function clearCache() { cache.clear(); }

export function debounce(fn, wait = 220) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

export const money = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);
export const pct = (v, digits = 1) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(digits)}%`);

export async function loadSql(panel, args = {}) {
  const body = await get(`/sql/${panel}`);
  return body.data.sql;
}

export function exportPanel(panel, params = {}) {
  const url = `/api/export/${panel}.csv${qs(params)}`;
  window.open(url, "_blank", "noopener");
}
