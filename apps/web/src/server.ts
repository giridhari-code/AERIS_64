/**
 * AERIS_64 TypeScript / Bun edge layer
 *
 * Python keeps the neural model (train / serve / memory).
 * This Bun service handles:
 *   - HTTP API gateway (proxy to Python NeuroField server)
 *   - Agent-facing helpers (git status, health)
 *   - Static UI (optional)
 *
 * Run:
 *   bun run dev          # http://localhost:3000
 *   AERIS_PYTHON_URL=http://127.0.0.1:8000 bun run start
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";
import { proxyChat, proxyHealth, pythonBase } from "./proxy";
import { gitStatus, gitLog, gitDiff } from "./git";

const app = new Hono();

app.use("*", logger());
app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "DELETE", "OPTIONS"],
  }),
);

app.get("/", (c) =>
  c.json({
    service: "aeris-web",
    runtime: "bun",
    language: "typescript",
    model_backend: pythonBase(),
    endpoints: {
      health: "GET /healthz",
      chat: "POST /v1/chat",
      git_status: "GET /v1/git/status",
      git_log: "GET /v1/git/log",
      git_diff: "GET /v1/git/diff",
    },
  }),
);

app.get("/healthz", async (c) => {
  const py = await proxyHealth();
  return c.json({
    web: "ok",
    runtime: "bun",
    python: py,
  });
});

/** Proxy chat to the Python NeuroField server */
app.post("/v1/chat", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const result = await proxyChat(body);
  return c.json(result.body, result.status as 200);
});

/** Local git helpers (same idea as the Python git_tool) */
app.get("/v1/git/status", async (c) => {
  const out = await gitStatus();
  return c.json(out);
});

app.get("/v1/git/log", async (c) => {
  const out = await gitLog();
  return c.json(out);
});

app.get("/v1/git/diff", async (c) => {
  const out = await gitDiff();
  return c.json(out);
});

const port = Number(process.env.PORT || 3000);
console.log(`AERIS web (Bun/TS) listening on http://0.0.0.0:${port}`);
console.log(`Python backend: ${pythonBase()}`);

export default {
  port,
  fetch: app.fetch,
};
