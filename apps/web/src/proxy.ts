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
  try {
    const res = await fetch(`${pythonBase()}/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
      signal: AbortSignal.timeout(120_000),
    });
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = JSON.parse(text);
    } catch {
      /* keep raw text */
    }
    return { status: res.status, body: parsed };
  } catch (e) {
    return {
      status: 502,
      body: {
        error: "python_backend_unreachable",
        detail: e instanceof Error ? e.message : String(e),
        hint: `Is the Python server running at ${pythonBase()}?`,
      },
    };
  }
}
