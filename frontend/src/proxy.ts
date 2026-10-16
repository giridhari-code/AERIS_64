/** Thin proxy to the Python NeuroField FastAPI server. */

const DEFAULT_PYTHON = "http://127.0.0.1:8000";

export function pythonBase(): string {
  return (
    process.env.AERIS_PYTHON_URL ||
    process.env.NEUROFIELD_URL ||
    DEFAULT_PYTHON
  ).replace(/\/$/, "");
}

export async function proxyHealth(): Promise<{
  ok: boolean;
  status?: number;
  detail?: string;
}> {
  try {
    const res = await fetch(`${pythonBase()}/healthz`, {
      signal: AbortSignal.timeout(3000),
    });
    return { ok: res.ok, status: res.status };
  } catch (e) {
    return {
      ok: false,
      detail: e instanceof Error ? e.message : String(e),
    };
  }
}

export async function proxyChat(body: unknown): Promise<{
  status: number;
  body: unknown;
}> {
  return proxyJson("POST", "/v1/chat", body);
}

/**
 * Streaming proxy (SSE pass-through). Returns the upstream body unbuffered so
 * `/v1/chat/stream` and `/ag-ui` stay real event streams through the edge.
 * Client disconnects propagate by cancelling the response body stream.
 */
export async function proxyStream(
  method: string,
  path: string,
  body?: unknown,
): Promise<Response> {
  try {
    const upstream = await fetch(`${pythonBase()}${path}`, {
      method,
      headers: {
        Accept: "text/event-stream",
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const contentType = upstream.headers.get("content-type") || "application/json";
    // Upstream error (or a non-streaming reply): pass it through as JSON, not as a broken stream.
    if (!upstream.ok || !contentType.includes("text/event-stream")) {
      const text = await upstream.text();
      return new Response(text, {
        status: upstream.status,
        headers: { "Content-Type": contentType },
      });
    }
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") {
      return new Response(null, { status: 499 });
    }
    return new Response(
      JSON.stringify({
        error: "python_backend_unreachable",
        detail: e instanceof Error ? e.message : String(e),
        hint: `Start Python model server at ${pythonBase()} first`,
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }
}

export async function proxyJson(
  method: string,
  path: string,
  body?: unknown,
): Promise<{ status: number; body: unknown }> {
  try {
    const res = await fetch(`${pythonBase()}${path}`, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(120_000),
    });
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = JSON.parse(text);
    } catch {
      /* keep raw */
    }
    return { status: res.status, body: parsed };
  } catch (e) {
    return {
      status: 502,
      body: {
        error: "python_backend_unreachable",
        detail: e instanceof Error ? e.message : String(e),
        hint: `Start Python model server at ${pythonBase()} first`,
      },
    };
  }
}
