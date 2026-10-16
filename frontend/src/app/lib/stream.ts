/**
 * Streaming chat transports, in preference order:
 *   1. /v1/ag-ui       — AG-UI protocol (spec 1.0) event stream
 *   2. /v1/chat/stream — plain SSE token stream
 *   3. /v1/chat        — buffered JSON (legacy fallback)
 *
 * A transport that is missing (404/501/…, e.g. ag-ui-protocol not installed)
 * or that does not answer with an event stream falls through to the next one,
 * so the UI keeps working against older model servers.
 */

import { authHeaders } from "./auth";

export type TransportName = "ag-ui" | "chat/stream" | "chat";

export interface ChatRequest {
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  autoLength: boolean;
  /** server-side session id (null = let the server create one) */
  sessionId: string | null;
  /** stable per-chat id, used as the AG-UI threadId (session continuity) */
  threadId: string;
}

export interface ChatMeta {
  sessionId: string | null;
  model?: string;
  latencyMs?: number;
  metaRag?: Record<string, unknown>;
}

export interface ChatOutcome {
  text: string;
  meta: ChatMeta;
  transport: TransportName;
}

export interface ChatHandlers {
  /** Incremental assistant text — render it live. */
  onDelta?: (chunk: string) => void;
}

/** The model/API answered with an error — show it, do not fall back. */
export class ChatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ChatError";
  }
}

/** Transport is not available; try the next one. */
class TransportUnavailable extends Error {
  constructor(
    public transport: TransportName,
    public status?: number,
    message?: string,
  ) {
    super(message || `transport ${transport} unavailable`);
    this.name = "TransportUnavailable";
  }
}

function isAbortError(e: unknown): boolean {
  return !!e && typeof e === "object" && (e as { name?: string }).name === "AbortError";
}

function isSse(res: Response): boolean {
  return (res.headers.get("content-type") || "").toLowerCase().includes("text/event-stream");
}

function apiHeaders(): Record<string, string> {
  return {
    ...authHeaders(),
    ...(window.__API_KEY ? { "X-API-Key": window.__API_KEY } : {}),
  };
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const text = await res.text();
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      const detail = j.detail;
      if (Array.isArray(detail)) {
        return detail
          .map((d) => (d as { msg?: string }).msg || JSON.stringify(d))
          .join("; ");
      }
      if (detail && typeof detail === "object") return JSON.stringify(detail);
      if (typeof detail === "string" && detail) return detail;
    } catch {
      /* not JSON */
    }
    return text.slice(0, 300) || res.statusText || String(res.status);
  } catch {
    return res.statusText || String(res.status);
  }
}

/** Parse an SSE body into `data:` payload strings. */
async function* sseData(res: Response): AsyncGenerator<string> {
  const body = res.body;
  if (!body) return;
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const flush = function* (block: string): Generator<string> {
    for (const line of block.split("\n")) {
      if (line.startsWith("data:")) yield line.slice(5).trimStart();
    }
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let idx = buf.indexOf("\n\n");
      while (idx >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        yield* flush(block);
        idx = buf.indexOf("\n\n");
      }
    }
    buf += decoder.decode().replace(/\r\n/g, "\n");
    yield* flush(buf);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}

function metaFromSentinel(ev: Record<string, unknown>, fallback: string | null): ChatMeta {
  const sentinel = ((ev.metadata as Record<string, unknown> | undefined)?.sentinel ?? {}) as Record<
    string,
    unknown
  >;
  return {
    sessionId: typeof sentinel.session_id === "string" ? sentinel.session_id : fallback,
    model: typeof sentinel.model === "string" ? sentinel.model : undefined,
    latencyMs: typeof sentinel.latency_ms === "number" ? sentinel.latency_ms : undefined,
    metaRag:
      sentinel.meta_rag && typeof sentinel.meta_rag === "object"
        ? (sentinel.meta_rag as Record<string, unknown>)
        : undefined,
  };
}

/* ------------------------- AG-UI (spec 1.0) ------------------------------ */

async function chatViaAgUi(
  req: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<ChatOutcome> {
  const runSuffix = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  const runId = "run-" + runSuffix;
  const res = await fetch("/v1/ag-ui", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...apiHeaders(),
    },
    body: JSON.stringify({
      threadId: req.threadId,
      runId,
      state: {},
      messages: [{ id: "user-" + runSuffix, role: "user", content: req.prompt }],
      tools: [],
      context: [],
      forwardedProps: {
        max_new_tokens: req.maxNewTokens,
        temperature: req.temperature,
        auto_length: req.autoLength,
      },
    }),
    signal,
  });
  if (res.status === 401) throw new ChatError(await errorDetail(res));
  if (!res.ok) throw new TransportUnavailable("ag-ui", res.status, await errorDetail(res));
  if (!isSse(res)) throw new TransportUnavailable("ag-ui", res.status);

  let text = "";
  let meta: ChatMeta = { sessionId: req.sessionId };
  let finished = false;
  for await (const data of sseData(res)) {
    if (data === "[DONE]") continue;
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(data) as Record<string, unknown>;
    } catch {
      continue;
    }
    switch (ev.type) {
      case "TEXT_MESSAGE_CONTENT": {
        const delta = typeof ev.delta === "string" ? ev.delta : "";
        if (delta) {
          text += delta;
          handlers.onDelta?.(delta);
        }
        break;
      }
      case "TEXT_MESSAGE_END":
        meta = metaFromSentinel(ev, meta.sessionId);
        break;
      case "RUN_ERROR":
        throw new ChatError(String(ev.message || "ag-ui run error"));
      case "RUN_FINISHED":
        finished = true;
        break;
      default:
        break;
    }
  }
  if (!finished && !text) throw new TransportUnavailable("ag-ui", res.status, "stream ended early");
  return { text, meta, transport: "ag-ui" };
}

/* --------------------------- plain SSE stream ---------------------------- */

async function chatViaStream(
  req: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<ChatOutcome> {
  const res = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...apiHeaders(),
    },
    body: JSON.stringify({
      prompt: req.prompt,
      max_new_tokens: req.maxNewTokens,
      temperature: req.temperature,
      auto_length: req.autoLength,
      session_id: req.sessionId,
    }),
    signal,
  });
  if (res.status === 401) throw new ChatError(await errorDetail(res));
  if (!res.ok) throw new TransportUnavailable("chat/stream", res.status, await errorDetail(res));
  if (!isSse(res)) throw new TransportUnavailable("chat/stream", res.status);

  let text = "";
  let meta: ChatMeta = { sessionId: req.sessionId };
  let done = false;
  for await (const data of sseData(res)) {
    if (data === "[DONE]") {
      done = true;
      continue;
    }
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(data) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (typeof ev.token === "string") {
      if (ev.token) {
        text += ev.token;
        handlers.onDelta?.(ev.token);
      }
    } else if (ev.event === "done") {
      done = true;
      meta = {
        sessionId: typeof ev.session_id === "string" ? ev.session_id : meta.sessionId,
        model: typeof ev.model === "string" ? ev.model : undefined,
        latencyMs: typeof ev.latency_ms === "number" ? ev.latency_ms : undefined,
        metaRag:
          ev.meta_rag && typeof ev.meta_rag === "object"
            ? (ev.meta_rag as Record<string, unknown>)
            : undefined,
      };
    } else if (ev.event === "error") {
      throw new ChatError(String(ev.detail || "stream error"));
    }
  }
  if (!done && !text) throw new TransportUnavailable("chat/stream", res.status, "stream ended early");
  return { text, meta, transport: "chat/stream" };
}

/* --------------------------- buffered fallback --------------------------- */

async function chatViaJson(
  req: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<ChatOutcome> {
  const res = await fetch("/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...apiHeaders() },
    body: JSON.stringify({
      prompt: req.prompt,
      max_new_tokens: req.maxNewTokens,
      temperature: req.temperature,
      auto_length: req.autoLength,
      session_id: req.sessionId,
    }),
    signal,
  });
  if (!res.ok) throw new ChatError(await errorDetail(res));
  const j = (await res.json()) as Record<string, unknown>;
  const text = typeof j.text === "string" ? j.text : "";
  if (text) handlers.onDelta?.(text);
  return {
    text,
    meta: {
      sessionId: typeof j.session_id === "string" ? j.session_id : req.sessionId,
      model: typeof j.model === "string" ? j.model : undefined,
      latencyMs: typeof j.latency_ms === "number" ? j.latency_ms : undefined,
      metaRag:
        j.meta_rag && typeof j.meta_rag === "object"
          ? (j.meta_rag as Record<string, unknown>)
          : undefined,
    },
    transport: "chat",
  };
}

/**
 * Send one turn, streaming when the backend supports it. Resolves with the
 * final (post-processed) assistant text and metadata.
 */
export async function chatWithStreaming(
  req: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<ChatOutcome> {
  const attempts = [chatViaAgUi, chatViaStream, chatViaJson];
  let lastUnavailable: TransportUnavailable | null = null;
  for (const attempt of attempts) {
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    try {
      return await attempt(req, handlers, signal);
    } catch (e) {
      if (isAbortError(e)) throw e;
      if (e instanceof TransportUnavailable) {
        lastUnavailable = e;
        continue;
      }
      throw e;
    }
  }
  throw new ChatError(lastUnavailable?.message || "no chat transport available");
}
