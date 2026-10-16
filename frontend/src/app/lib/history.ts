/**
 * Multi-chat history persisted in localStorage (`sentinel_chats_v1`).
 * The legacy `sentinel_session` key stays in sync with the active chat so old
 * tooling (and the jsdom e2e checks) keep working.
 */
import type { ChatMessage } from "../components/Message";

export interface StoredChat {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  sessionId: string | null;
  messages: ChatMessage[];
  /** Who owns this chat: "guest" or an account id (`session.user.id`). */
  owner: string;
}

export interface ChatStore {
  version: 1;
  currentId: string;
  chats: StoredChat[];
}

const KEY = "sentinel_chats_v1";
export const MAX_CHATS = 30;
const MAX_MESSAGES = 300;
export const DEFAULT_TITLE = "New chat";

export function newChatId(): string {
  return "chat-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
}

export function freshChat(owner = "guest", now = Date.now()): StoredChat {
  return {
    id: newChatId(),
    title: DEFAULT_TITLE,
    createdAt: now,
    updatedAt: now,
    sessionId: null,
    messages: [],
    owner,
  };
}

/** Make sure `owner` has a visible current chat (login/logout switches). */
export function ensureOwnerChat(s: ChatStore, owner: string): ChatStore {
  const mine = s.chats.filter((c) => c.owner === owner);
  if (mine.some((c) => c.id === s.currentId)) return s;
  if (mine.length) {
    const next = [...mine].sort((a, b) => b.updatedAt - a.updatedAt)[0]!;
    return { ...s, currentId: next.id };
  }
  const c = freshChat(owner);
  return { ...s, currentId: c.id, chats: [c, ...s.chats].slice(0, MAX_CHATS) };
}

export function titleFrom(text: string): string {
  const t = (text || "").replace(/\s+/g, " ").trim();
  if (!t) return DEFAULT_TITLE;
  return t.length > 48 ? t.slice(0, 48).trimEnd() + "…" : t;
}

export function relTime(ts: number, now = Date.now()): string {
  const s = Math.max(0, Math.floor((now - ts) / 1000));
  if (s < 45) return "just now";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  if (s < 7 * 86400) return Math.floor(s / 86400) + "d ago";
  try {
    return new Date(ts).toLocaleDateString();
  } catch {
    return "";
  }
}

function sanitizeMessage(m: unknown): ChatMessage | null {
  if (!m || typeof m !== "object") return null;
  const o = m as Record<string, unknown>;
  const role = o.role === "user" || o.role === "assistant" ? o.role : null;
  if (!role || typeof o.text !== "string") return null;
  const id = typeof o.id === "number" && Number.isFinite(o.id) ? o.id : 0;
  return {
    id,
    role,
    text: o.text,
    meta: typeof o.meta === "string" ? o.meta : undefined,
  };
}

function sanitizeChat(c: unknown): StoredChat | null {
  if (!c || typeof c !== "object") return null;
  const o = c as Record<string, unknown>;
  if (typeof o.id !== "string" || !o.id) return null;
  const now = Date.now();
  const messages = Array.isArray(o.messages)
    ? (o.messages.map(sanitizeMessage).filter(Boolean) as ChatMessage[])
    : [];
  return {
    id: o.id,
    title: typeof o.title === "string" && o.title ? o.title : DEFAULT_TITLE,
    createdAt: typeof o.createdAt === "number" ? o.createdAt : now,
    updatedAt: typeof o.updatedAt === "number" ? o.updatedAt : now,
    sessionId: typeof o.sessionId === "string" ? o.sessionId : null,
    messages,
    owner: typeof o.owner === "string" && o.owner ? o.owner : "guest",
  };
}

export function loadStore(): ChatStore {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || "null") as Record<string, unknown> | null;
    if (raw && Array.isArray(raw.chats)) {
      const chats = raw.chats.map(sanitizeChat).filter(Boolean) as StoredChat[];
      if (chats.length) {
        const currentId =
          typeof raw.currentId === "string" && chats.some((c) => c.id === raw.currentId)
            ? raw.currentId
            : chats[0]!.id;
        return { version: 1, currentId, chats: chats.slice(0, MAX_CHATS) };
      }
    }
  } catch {
    /* corrupt store — start over */
  }
  // Migrate the legacy single-session key into the first chat.
  let sessionId: string | null = null;
  try {
    sessionId = localStorage.getItem("sentinel_session");
  } catch {
    /* ignore */
  }
  const c = freshChat();
  c.sessionId = sessionId;
  return { version: 1, currentId: c.id, chats: [c] };
}

export function persistStore(s: ChatStore): void {
  const chats = s.chats.slice(0, MAX_CHATS).map((c) => ({
    ...c,
    messages: c.messages.slice(-MAX_MESSAGES),
  }));
  try {
    localStorage.setItem(KEY, JSON.stringify({ version: 1, currentId: s.currentId, chats }));
  } catch {
    // Quota exceeded — retry with message bodies trimmed from non-current chats.
    try {
      const lean = chats.map((c) =>
        c.id === s.currentId ? c : { ...c, messages: c.messages.slice(-5) },
      );
      localStorage.setItem(KEY, JSON.stringify({ version: 1, currentId: s.currentId, chats: lean }));
    } catch {
      /* give up silently */
    }
  }
}
