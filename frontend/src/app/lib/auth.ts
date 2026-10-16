/**
 * Local account sessions for the web client.
 *
 * Talks to the Python backend's `/v1/auth/*` routes (stdlib sqlite store —
 * see `packages/sentinel/serving/accounts.py`). The session (bearer token +
 * user) lives in localStorage under `sentinel_auth_v1`.
 *
 * If the auth endpoints are missing (older servers, offline dev, jsdom tests)
 * the app silently falls back to guest mode — chat must never be blocked by
 * auth.
 */

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  created_at?: number;
}

export type Session =
  | { kind: "user"; token: string; user: AuthUser }
  | { kind: "guest" };

export interface AuthStatus {
  accounts: boolean;
  guest_ok: boolean;
  required: boolean;
  users?: number;
}

const KEY = "sentinel_auth_v1";

let cache: Session | null = null;
let cacheLoaded = false;

function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** Read the persisted session (synchronous — safe to call during render). */
export function loadSession(): Session | null {
  if (cacheLoaded) return cache;
  cacheLoaded = true;
  cache = null;
  try {
    const raw = storage()?.getItem(KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Session;
    if (s && s.kind === "guest") {
      cache = { kind: "guest" };
    } else if (
      s &&
      s.kind === "user" &&
      typeof s.token === "string" &&
      s.token &&
      s.user &&
      typeof s.user.id === "string" &&
      typeof s.user.name === "string" &&
      typeof s.user.email === "string"
    ) {
      cache = {
        kind: "user",
        token: s.token,
        user: { id: s.user.id, name: s.user.name, email: s.user.email, created_at: s.user.created_at },
      };
    }
  } catch {
    cache = null;
  }
  return cache;
}

/** Persist (or clear, with null) the session. */
export function saveSession(s: Session | null): void {
  cacheLoaded = true;
  cache = s;
  try {
    if (s) storage()?.setItem(KEY, JSON.stringify(s));
    else storage()?.removeItem(KEY);
  } catch {
    /* private mode / quota — session simply won't survive a reload */
  }
}

/** `Authorization` header for model calls (empty for guests). */
export function authHeaders(): Record<string, string> {
  const s = loadSession();
  return s && s.kind === "user" ? { Authorization: `Bearer ${s.token}` } : {};
}

/** `AbortSignal.timeout` when available (all modern browsers); else no signal. */
function timeoutSignal(ms: number): AbortSignal | undefined {
  const AS = AbortSignal as unknown as { timeout?: (m: number) => AbortSignal };
  if (typeof AS.timeout === "function") {
    try {
      return AS.timeout(ms);
    } catch {
      /* fall through */
    }
  }
  return undefined;
}

async function detail(res: Response): Promise<string> {
  try {
    const text = await res.text();
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      const d = j.detail;
      if (Array.isArray(d)) {
        return (
          d.map((x) => (x as { msg?: string }).msg || JSON.stringify(x)).join("; ") ||
          `HTTP ${res.status}`
        );
      }
      if (typeof d === "string" && d) return d;
      if (d && typeof d === "object") return JSON.stringify(d);
    } catch {
      /* not JSON */
    }
    return text.slice(0, 300) || res.statusText || `HTTP ${res.status}`;
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

function parseAuthPayload(j: unknown): Session {
  const o = (j || {}) as Record<string, unknown>;
  const token = typeof o.token === "string" ? o.token : "";
  const u = (o.user || {}) as Record<string, unknown>;
  if (!token || typeof u.id !== "string" || typeof u.name !== "string" || typeof u.email !== "string") {
    throw new Error("malformed auth response from server");
  }
  return {
    kind: "user",
    token,
    user: {
      id: u.id,
      name: u.name,
      email: u.email,
      created_at: typeof u.created_at === "number" ? u.created_at : undefined,
    },
  };
}

/** Public capability probe. Returns null when auth is unavailable/disabled. */
export async function authStatus(): Promise<AuthStatus | null> {
  try {
    const res = await fetch("/v1/auth/status", {
      headers: { Accept: "application/json" },
      signal: timeoutSignal(8000),
    });
    if (!res.ok) return null;
    const j = (await res.json()) as Partial<AuthStatus>;
    if (j && j.accounts === true) {
      return {
        accounts: true,
        guest_ok: j.guest_ok !== false,
        required: j.required === true,
        users: typeof j.users === "number" ? j.users : undefined,
      };
    }
    return null;
  } catch {
    return null;
  }
}

async function postAuth(path: string, body: unknown): Promise<Session> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
    signal: timeoutSignal(20000),
  });
  if (!res.ok) throw new Error(await detail(res));
  return parseAuthPayload(await res.json());
}

export async function signup(name: string, email: string, password: string): Promise<Session> {
  return postAuth("/v1/auth/signup", { name, email, password });
}

export async function login(email: string, password: string): Promise<Session> {
  return postAuth("/v1/auth/login", { email, password });
}

/** Best-effort server-side revocation; local state is cleared by the caller. */
export async function logout(s: Session): Promise<void> {
  if (s.kind !== "user") return;
  try {
    await fetch("/v1/auth/logout", {
      method: "POST",
      headers: { Authorization: `Bearer ${s.token}` },
      signal: timeoutSignal(8000),
    });
  } catch {
    /* offline logout is fine — token expires server-side */
  }
}

/** Check a stored token: "invalid" only on a definitive 401 (clear it then). */
export async function validateSession(s: Session): Promise<"ok" | "invalid" | "unknown"> {
  if (s.kind !== "user") return "ok";
  try {
    const res = await fetch("/v1/auth/me", {
      headers: { Authorization: `Bearer ${s.token}` },
      signal: timeoutSignal(8000),
    });
    if (res.ok) return "ok";
    if (res.status === 401) return "invalid";
    return "unknown";
  } catch {
    return "unknown";
  }
}
