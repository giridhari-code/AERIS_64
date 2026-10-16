/**
 * AERIS_64 TypeScript / Bun edge layer — real frontend + API gateway
 *
 * Python  (:8000) → neural model only
 * Bun     (:3000) → chat UI + proxy + git helpers
 *
 *   bun run dev
 *   AERIS_PYTHON_URL=http://127.0.0.1:8000 bun run start
 */

import { Hono } from "hono";
import type { Context } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";
import { serveStatic } from "hono/bun";
import { proxyChat, proxyHealth, proxyJson, proxyStream, pythonBase } from "./proxy";
import { gitStatus, gitLog, gitDiff } from "./git";
import { buildAppBundle, latestAppMtime } from "./app-build";

const app = new Hono();

app.use("*", logger());
app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
  }),
);

/** Forward the browser's Authorization header (accounts bearer token). */
function bearerOf(c: Context): Record<string, string> {
  const h = c.req.header("authorization");
  return h ? { Authorization: h } : {};
}


// ---- Edge rate limit (per IP, in-memory) ----------------------------------
const RATE_MAX = Number(process.env.AERIS_RATE_LIMIT || 60);
const RATE_WINDOW_MS = Number(process.env.AERIS_RATE_WINDOW_MS || 60_000);
const hits = new Map<string, number[]>();

function rateLimit(ip: string): boolean {
  const now = Date.now();
  const q = (hits.get(ip) || []).filter((t) => now - t < RATE_WINDOW_MS);
  if (q.length >= RATE_MAX) {
    hits.set(ip, q);
    return false;
  }
  q.push(now);
  hits.set(ip, q);
  if (hits.size > 10_000) {
    // crude GC
    for (const [k, v] of hits) {
      const last = v.at(-1);
      if (last === undefined || now - last > RATE_WINDOW_MS) hits.delete(k);
    }
  }
  return true;
}

app.use("*", async (c, next) => {
  const path = c.req.path || "";
  if (path.startsWith("/v1/") || path === "/ag-ui" || path === "/healthz" || path === "/readyz") {
    const ip =
      c.req.header("x-forwarded-for")?.split(",")[0]?.trim() ||
      c.req.header("x-real-ip") ||
      "anon";
    if ((path.startsWith("/v1/") || path === "/ag-ui") && !rateLimit(ip)) {
      return c.json({ detail: "Rate limit exceeded" }, 429);
    }
  }
  await next();
});

// ---- API (proxied to Python model server) --------------------------------

app.get("/healthz", async (c) => {
  const py = await proxyHealth();
  return c.json({ web: "ok", runtime: "bun", python: py });
});

app.get("/readyz", async (c) => {
  const r = await proxyJson("GET", "/readyz");
  return c.json(r.body, r.status as 200);
});

app.post("/v1/chat", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const result = await proxyChat(body, bearerOf(c));
  return c.json(result.body, result.status as 200);
});

// SSE pass-through (no buffering): token deltas reach the browser live.
app.post("/v1/chat/stream", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  return proxyStream("POST", "/v1/chat/stream", body, bearerOf(c));
});

// AG-UI protocol (spec 1.0) — agent-UI event stream, proxied unbuffered.
app.post("/ag-ui", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  return proxyStream("POST", "/ag-ui", body, bearerOf(c));
});

app.post("/v1/ag-ui", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  return proxyStream("POST", "/v1/ag-ui", body, bearerOf(c));
});

app.post("/v1/reset", async (c) => {
  const url = new URL(c.req.url);
  const r = await proxyJson("POST", `/v1/reset${url.search}`, undefined, bearerOf(c));
  return c.json(r.body, r.status as 200);
});

app.get("/v1/models", async (c) => {
  const r = await proxyJson("GET", "/v1/models");
  return c.json(r.body, r.status as 200);
});

// ---- Accounts: signup / login / me / logout (proxied to the Python store) --

app.get("/v1/auth/status", async (c) => {
  const r = await proxyJson("GET", "/v1/auth/status");
  return c.json(r.body, r.status as 200);
});

app.post("/v1/auth/signup", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/auth/signup", body);
  return c.json(r.body, r.status as 200);
});

app.post("/v1/auth/login", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/auth/login", body);
  return c.json(r.body, r.status as 200);
});

app.get("/v1/auth/me", async (c) => {
  const r = await proxyJson("GET", "/v1/auth/me", undefined, bearerOf(c));
  return c.json(r.body, r.status as 200);
});

app.post("/v1/auth/logout", async (c) => {
  const r = await proxyJson("POST", "/v1/auth/logout", undefined, bearerOf(c));
  return c.json(r.body, r.status as 200);
});

// ---- Git helpers (local, no Python needed) --------------------------------

app.get("/v1/git/status", async (c) => c.json(await gitStatus()));
app.get("/v1/git/log", async (c) => c.json(await gitLog()));
app.get("/v1/git/diff", async (c) => c.json(await gitDiff()));

// ---- Product APIs (proxy to Python: kalofacts / ajeebwork / SI_v1) --------

app.get("/v1/kalofacts", async (c) => {
  const r = await proxyJson("GET", "/v1/kalofacts");
  return c.json(r.body, r.status as 200);
});

app.post("/v1/kalofacts", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/kalofacts", body);
  return c.json(r.body, r.status as 200);
});

app.get("/v1/kalofacts/:id", async (c) => {
  const r = await proxyJson("GET", `/v1/kalofacts/${c.req.param("id")}`);
  return c.json(r.body, r.status as 200);
});

app.put("/v1/kalofacts/:id", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("PUT", `/v1/kalofacts/${c.req.param("id")}`, body);
  return c.json(r.body, r.status as 200);
});

app.delete("/v1/kalofacts/:id", async (c) => {
  const r = await proxyJson("DELETE", `/v1/kalofacts/${c.req.param("id")}`);
  return c.json(r.body, r.status as 200);
});

app.post("/v1/ajeebwork/plan", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/ajeebwork/plan", body);
  return c.json(r.body, r.status as 200);
});

app.post("/v1/si/v1", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/si/v1", body);
  return c.json(r.body, r.status as 200);
});

// ---- Buddy companion (float UI; proxy to Python) -------------------------
app.get("/v1/buddy", async (c) => {
  const q = c.req.query("user_id") || "";
  const path = q ? `/v1/buddy?user_id=${encodeURIComponent(q)}` : "/v1/buddy";
  const r = await proxyJson("GET", path);
  return c.json(r.body, r.status as 200);
});

app.post("/v1/buddy", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/buddy", body);
  return c.json(r.body, r.status as 200);
});

// Egg incubation: the chat UI posts one step per user message
app.post("/v1/buddy/step", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/buddy/step", body);
  return c.json(r.body, r.status as 200);
});

// First login: carry the guest (device) egg/buddy into the account
app.post("/v1/buddy/adopt", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/buddy/adopt", body);
  return c.json(r.body, r.status as 200);
});

// ---- React app bundle (built in memory; rebuilt when src/app changes) ------

let APP_JS = "";
let APP_OK = false;
let APP_BUILD_AT = 0;

async function ensureAppBundle(force = false): Promise<void> {
  if (!force && APP_OK && latestAppMtime() <= APP_BUILD_AT) return;
  const r = await buildAppBundle();
  if (r.ok) {
    APP_JS = r.js;
    APP_OK = true;
    APP_BUILD_AT = Date.now();
  } else if (!APP_JS) {
    APP_OK = false;
  }
}
await ensureAppBundle(true);

app.get("/static/app.js", async (c) => {
  await ensureAppBundle();
  if (!APP_OK) {
    return c.body("/* app bundle failed to build — check `bun run dev` logs */", 500, {
      "Content-Type": "text/javascript; charset=utf-8",
    });
  }
  return c.body(APP_JS, 200, {
    "Content-Type": "text/javascript; charset=utf-8",
    "Cache-Control": "no-store",
  });
});

// ---- Static chat UI (same look as Python frontend) ------------------------

app.use("/static/*", serveStatic({ root: "./public" }));
app.get("/logo.svg", serveStatic({ path: "./public/logo.svg" }));

// Serve chat UI at /
app.get("/", serveStatic({ path: "./public/index.html" }));

app.get("/product", serveStatic({ path: "./public/product.html" }));

// Fallback: any unknown GET → UI
app.get("*", serveStatic({ path: "./public/index.html" }));

const port = Number(process.env.PORT || 3000);
console.log(`AERIS frontend (Bun/TS)  → http://0.0.0.0:${port}`);
console.log(`Python model backend     → ${pythonBase()}`);

export default {
  port,
  fetch: app.fetch,
  // Max idle window for sockets/streams. Needed for SSE: the demo model can
  // take many seconds between tokens, and Bun's default (10s) would ECONNRESET
  // slow streams mid-generation.
  idleTimeout: 255,
};
