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
import { cors } from "hono/cors";
import { logger } from "hono/logger";
import { serveStatic } from "hono/bun";
import { proxyChat, proxyHealth, proxyJson, pythonBase } from "./proxy";
import { gitStatus, gitLog, gitDiff } from "./git";

const app = new Hono();

app.use("*", logger());
app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  }),
);

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
  const result = await proxyChat(body);
  return c.json(result.body, result.status as 200);
});

app.post("/v1/chat/stream", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const r = await proxyJson("POST", "/v1/chat/stream", body);
  return c.json(r.body, r.status as 200);
});

app.post("/v1/reset", async (c) => {
  const url = new URL(c.req.url);
  const r = await proxyJson("POST", `/v1/reset${url.search}`);
  return c.json(r.body, r.status as 200);
});

app.get("/v1/models", async (c) => {
  const r = await proxyJson("GET", "/v1/models");
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
};
