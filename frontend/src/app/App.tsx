import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Topbar, type Health } from "./components/Topbar";
import { Welcome } from "./components/Welcome";
import { Message, TypingIndicator, type ChatMessage } from "./components/Message";
import { Composer } from "./components/Composer";
import { BuddyFloat } from "./components/BuddyFloat";
import { ModeSheet, type Effort, type ContextMode } from "./components/ModeSheet";
import { Sidebar } from "./components/Sidebar";
import { AuthGate } from "./components/AuthGate";
import { initBuddy } from "./lib/buddy-float";
import { buddyUserId, matchBuddyOnOff } from "../buddy";
import { ChatError, chatWithStreaming, type ChatRequest } from "./lib/stream";
import {
  authStatus as fetchAuthStatus,
  loadSession,
  logout,
  saveSession,
  validateSession,
  type AuthStatus,
  type Session,
} from "./lib/auth";
import {
  DEFAULT_TITLE,
  ensureOwnerChat,
  freshChat,
  loadStore,
  MAX_CHATS,
  persistStore,
  titleFrom,
  type ChatStore,
} from "./lib/history";

const PROMPT_PLACEHOLDER = "Message SENTINEL…";

interface Settings {
  maxTok: string;
  temp: string;
  autoLen: boolean;
  effort: Effort;
  context: ContextMode;
}

function loadSettings(): Settings {
  try {
    const s = JSON.parse(localStorage.getItem("sentinel_ui") || "{}");
    return {
      maxTok: s.maxTok != null ? String(s.maxTok) : "64",
      temp: s.temp != null ? String(s.temp) : "0.4",
      autoLen: s.autoLen != null ? !!s.autoLen : true,
      effort: (s.effort as Effort) || "standard",
      context: (s.context as ContextMode) || "standard",
    };
  } catch (_) {
    return { maxTok: "64", temp: "0.4", autoLen: true, effort: "standard", context: "standard" };
  }
}

function fmtMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "";
  if (ms < 1000) return Math.round(ms) + " ms";
  return (ms / 1000).toFixed(1) + " s";
}

function isAbort(e: unknown): boolean {
  return !!e && typeof e === "object" && (e as { name?: string }).name === "AbortError";
}

function isDesktop(): boolean {
  try {
    return window.matchMedia("(min-width: 901px)").matches;
  } catch (_) {
    return true;
  }
}

export function App() {
  /* Monotonic message id across the whole session — seeded once from the store
     so ids stay unique even if messages arrive before effects flush. */
  const idRef = useRef(0);
  const [store, setStore] = useState<ChatStore>(() => {
    const s = loadStore();
    let max = 0;
    for (const c of s.chats) for (const m of c.messages) if (m.id > max) max = m.id;
    idRef.current = max;
    return s;
  });
  const [typing, setTyping] = useState(false);
  const [busy, setBusy] = useState(false);
  const [latencyText, setLatencyText] = useState("");
  const [health, setHealth] = useState<Health>({ state: "unknown", label: "…" });
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [modeOpen, setModeOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(isDesktop);
  const [session, setSession] = useState<Session | null>(() => loadSession());
  /** "checking" while probing; null = auth unavailable (auto-guest fallback). */
  const [authStatus, setAuthStatus] = useState<AuthStatus | "checking" | null>("checking");

  /** Account identity for chat ownership ("guest" = this device only). */
  const owner = session?.kind === "user" ? session.user.id : "guest";

  const active = useMemo(() => {
    const mine = store.chats.filter((c) => c.owner === owner);
    return mine.find((c) => c.id === store.currentId) || mine[0] || store.chats[0]!;
  }, [store, owner]);
  const visibleChats = useMemo(
    () => store.chats.filter((c) => c.owner === owner),
    [store, owner],
  );
  const messages = active.messages;

  const promptRef = useRef<HTMLTextAreaElement | null>(null);
  const chatRef = useRef<HTMLElement | null>(null);
  const busyRef = useRef(false);
  const epochRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const storeRef = useRef(store);
  storeRef.current = store;
  const sessionIdRef = useRef<string | null>(active.sessionId);
  sessionIdRef.current = active.sessionId;

  /* Persist the chat store (debounced) */
  useEffect(() => {
    const t = window.setTimeout(() => persistStore(store), 250);
    return () => window.clearTimeout(t);
  }, [store]);

  useEffect(() => {
    const save = () => persistStore(storeRef.current);
    window.addEventListener("beforeunload", save);
    window.addEventListener("pagehide", save);
    return () => {
      window.removeEventListener("beforeunload", save);
      window.removeEventListener("pagehide", save);
    };
  }, []);

  /* Legacy key stays in sync with the active chat's server session */
  useEffect(() => {
    try {
      if (active.sessionId) localStorage.setItem("sentinel_session", active.sessionId);
      else localStorage.removeItem("sentinel_session");
    } catch (_) {}
  }, [active.sessionId]);

  const setBusyBoth = useCallback((v: boolean) => {
    busyRef.current = v;
    setBusy(v);
  }, []);

  const updateActive = useCallback((fn: (c: ChatStore["chats"][number]) => ChatStore["chats"][number]) => {
    setStore((s) => ({
      ...s,
      chats: s.chats.map((c) => (c.id === s.currentId ? fn(c) : c)),
    }));
  }, []);

  const appendMessage = useCallback(
    (role: ChatMessage["role"], text: string, meta?: string) => {
      const id = ++idRef.current; // outside the updater — updaters must stay pure
      updateActive((c) => {
        const msg: ChatMessage = { id, role, text, meta };
        const title =
          c.title === DEFAULT_TITLE && role === "user" && !c.messages.length
            ? titleFrom(text)
            : c.title;
        return { ...c, title, updatedAt: Date.now(), messages: [...c.messages, msg] };
      });
    },
    [updateActive],
  );

  /** Streaming: extend the live assistant bubble, or start a new one. */
  const appendDelta = useCallback(
    (delta: string) => {
      const newId = ++idRef.current; // outside the updater — updaters must stay pure
      updateActive((c) => {
        const last = c.messages[c.messages.length - 1];
        if (last && last.role === "assistant" && last.streaming) {
          const next: ChatMessage = { ...last, text: last.text + delta };
          return { ...c, messages: [...c.messages.slice(0, -1), next], updatedAt: Date.now() };
        }
        const msg: ChatMessage = { id: newId, role: "assistant", text: delta, streaming: true };
        return { ...c, messages: [...c.messages, msg], updatedAt: Date.now() };
      });
    },
    [updateActive],
  );

  /** Finish the live assistant bubble (text null = keep whatever streamed). */
  const finalizeAssistant = useCallback(
    (text: string | null, meta?: string) => {
      updateActive((c) => {
        const last = c.messages[c.messages.length - 1];
        if (!last || last.role !== "assistant" || !last.streaming) return c;
        const next: ChatMessage = {
          ...last,
          text: text != null ? text : last.text,
          streaming: false,
          meta: meta ?? last.meta,
        };
        return { ...c, messages: [...c.messages.slice(0, -1), next], updatedAt: Date.now() };
      });
    },
    [updateActive],
  );

  const autoGrow = useCallback(() => {
    const el = promptRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, []);

  const resetServerSession = useCallback((sid: string | null) => {
    if (!sid) return;
    fetch("/v1/reset?session_id=" + encodeURIComponent(sid), { method: "POST" }).catch(() => {});
  }, []);

  /** Drop the active chat's server session (client + server). */
  const forgetSession = useCallback(() => {
    resetServerSession(sessionIdRef.current);
    sessionIdRef.current = null;
  }, [resetServerSession]);

  /** Cancel any in-flight generation; further async updates are epoch-dropped. */
  const cancelInFlight = useCallback(() => {
    epochRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    busyRef.current = false;
    setBusy(false);
    setTyping(false);
  }, []);

  /* ---- accounts (login / signup / guest) -------------------------------- */

  const probeAuth = useCallback(async () => {
    setAuthStatus("checking");
    const st = await fetchAuthStatus();
    setAuthStatus(st);
    // Auth unavailable (older server / offline): fall back to a transient
    // guest session so chat keeps working. Not persisted — a later reload
    // against a real auth backend shows the gate again.
    if (!st || !st.accounts) setSession({ kind: "guest" });
  }, []);

  useEffect(() => {
    const s = loadSession();
    if (!s) {
      void probeAuth();
      return;
    }
    setSession(s);
    setAuthStatus(null);
    if (s.kind === "user") {
      void validateSession(s).then((v) => {
        if (v === "invalid") {
          // Expired/revoked token — clear it and fall back to the gate.
          saveSession(null);
          setSession(null);
          void probeAuth();
        }
      });
    }
  }, [probeAuth]);

  /* Belt & braces: whenever the owner changes, keep a visible chat selected. */
  useEffect(() => {
    setStore((prev) => ensureOwnerChat(prev, owner));
  }, [owner]);

  const enterSession = useCallback(
    (s: Session) => {
      saveSession(s);
      cancelInFlight();
      setStore((prev) => ensureOwnerChat(prev, s.kind === "user" ? s.user.id : "guest"));
      setSession(s);
      setAuthStatus(null);
    },
    [cancelInFlight],
  );

  const handleLogout = useCallback(() => {
    const s = loadSession();
    if (s && s.kind === "user") void logout(s);
    cancelInFlight();
    saveSession(null);
    setStore((prev) => ensureOwnerChat(prev, "guest"));
    setSession(null);
    void probeAuth();
  }, [cancelInFlight, probeAuth]);

  const requestLogin = useCallback(() => {
    cancelInFlight();
    saveSession(null);
    setSession(null);
    void probeAuth();
  }, [cancelInFlight, probeAuth]);

  /* Buddy float (imperative controller — same DOM ids/behavior as the legacy UI).
     Init once a chat session exists: during the boot splash / auth gate the
     float is not mounted yet and the controller binds its DOM exactly once. */
  useEffect(() => {
    if (session) initBuddy();
  }, [session]);

  /* Backend health poll */
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const r = await fetch("/readyz");
        if (!r.ok) throw new Error("not ready");
        let label = "online";
        try {
          const j = await r.json();
          if (j.model) label = j.model;
          else if (j.python && j.python.model) label = j.python.model;
          else if (j.status) label = j.status;
        } catch (_) {}
        if (alive) setHealth({ state: "ok", label });
      } catch (_) {
        if (alive) setHealth({ state: "err", label: "offline" });
      }
    };
    check();
    const t = window.setInterval(check, 30000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);

  /* Keep the thread scrolled to the newest content */
  useEffect(() => {
    const chat = chatRef.current;
    if (chat) chat.scrollTop = chat.scrollHeight;
  }, [messages, typing]);

  const updateSettings = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        localStorage.setItem(
          "sentinel_ui",
          JSON.stringify({ maxTok: next.maxTok, temp: next.temp, autoLen: next.autoLen, effort: next.effort, context: next.context }),
        );
      } catch (_) {}
      return next;
    });
  }, []);

  /* ---- chat history (sidebar) ------------------------------------------- */

  const newChat = useCallback(() => {
    forgetSession();
    cancelInFlight();
    setLatencyText("");
    setEditingId(null);
    setStore((s) => {
      const c = freshChat(owner);
      const keep = s.chats.filter((x) => x.id !== s.currentId || x.messages.length > 0);
      return { version: 1, currentId: c.id, chats: [c, ...keep].slice(0, MAX_CHATS) };
    });
    if (!isDesktop()) setSidebarOpen(false);
  }, [cancelInFlight, forgetSession, owner]);

  const clearChat = useCallback(() => {
    forgetSession();
    cancelInFlight();
    setLatencyText("");
    setEditingId(null);
    updateActive((c) => ({
      ...c,
      messages: [],
      sessionId: null,
      title: DEFAULT_TITLE,
      updatedAt: Date.now(),
    }));
  }, [cancelInFlight, forgetSession, updateActive]);

  const selectChat = useCallback(
    (id: string) => {
      if (id !== storeRef.current.currentId) {
        cancelInFlight();
        setEditingId(null);
        setLatencyText("");
        setStore((s) => ({ ...s, currentId: id }));
      }
      if (!isDesktop()) setSidebarOpen(false);
    },
    [cancelInFlight],
  );

  const deleteChat = useCallback(
    (id: string) => {
      if (id === storeRef.current.currentId) {
        cancelInFlight();
        setEditingId(null);
        setLatencyText("");
      }
      setStore((s) => {
        const chats = s.chats.filter((c) => c.id !== id);
        if (!chats.length) {
          const c = freshChat(owner);
          return { version: 1, currentId: c.id, chats: [c] };
        }
        return ensureOwnerChat(
          { version: 1, currentId: s.currentId === id ? "" : s.currentId, chats },
          owner,
        );
      });
    },
    [cancelInFlight, owner],
  );

  /* ---- generate / stop / edit / regenerate ------------------------------ */

  const runPrompt = useCallback(
    async (
      text: string,
      opts: { appendUser: boolean; sessionId: string | null; threadId: string },
    ) => {
      const epoch = epochRef.current;
      if (opts.appendUser) appendMessage("user", text);
      setBusyBoth(true);
      setTyping(true);
      try {
        window.sentinelBuddyReact?.(text, "user");
      } catch (_) {}

      const ac = new AbortController();
      abortRef.current = ac;
      let streamed = false;

      const onDelta = (delta: string) => {
        if (epoch !== epochRef.current || !delta) return;
        if (!streamed) {
          streamed = true;
          setTyping(false);
        }
        appendDelta(delta);
      };

      const maxTok = parseInt(settings.maxTok, 10) || 64;
      const temp = parseFloat(settings.temp);
      const req: ChatRequest = {
        prompt: text,
        maxNewTokens: maxTok,
        temperature: Number.isFinite(temp) ? temp : 0.4,
        autoLength: settings.autoLen,
        sessionId: opts.sessionId,
        threadId: opts.threadId,
        userId: buddyUserId(),
      };

      try {
        const outcome = await chatWithStreaming(req, { onDelta }, ac.signal);
        if (epoch !== epochRef.current) return; // chat switched / reset while waiting — drop
        setTyping(false);
        const reply = (outcome.text || "").trim() || "(empty reply — train with more SFT data)";
        const lat = outcome.meta.latencyMs != null ? fmtMs(outcome.meta.latencyMs) : "";
        const meta = lat ? "latency " + lat : undefined;
        if (streamed) finalizeAssistant(reply, meta);
        else appendMessage("assistant", reply, meta);

        if (outcome.meta.sessionId) {
          const sid = outcome.meta.sessionId;
          sessionIdRef.current = sid;
          updateActive((c) => ({ ...c, sessionId: sid }));
        }
        try {
          window.sentinelBuddyReact?.(reply, "assistant");
        } catch (_) {}
        const bmeta = outcome.meta.metaRag || {};
        if (bmeta.buddy_egg || reply === "BUDDY_EGG_UI") {
          try {
            window.sentinelBuddySetVisible?.(true);
            window.sentinelBuddyStep?.();
          } catch (_) {}
          if (reply === "BUDDY_EGG_UI") {
            appendMessage("assistant", "Egg incubating — progress is on the Buddy float (not in this chat).");
            setBusy(false);
            return;
          }
        }
        if (bmeta.buddy_ui === "on" || bmeta.buddy_ui === "off") {
          try {
            window.sentinelBuddySetVisible?.(bmeta.buddy_ui === "on");
          } catch (_) {}
        } else if (bmeta.buddy && /♥/.test(reply)) {
          try {
            window.sentinelBuddyPet?.();
          } catch (_) {}
        }
        setLatencyText(lat ? "last " + lat : "");
      } catch (e) {
        if (epoch !== epochRef.current) return;
        setTyping(false);
        if (isAbort(e)) {
          // Stopped by the user — keep whatever text streamed so far.
          finalizeAssistant(null, "stopped");
        } else if (e instanceof ChatError) {
          appendMessage("assistant", "Error: " + e.message);
        } else {
          appendMessage("assistant", "Network error: " + (e as Error).message);
        }
      } finally {
        if (epoch === epochRef.current) setBusyBoth(false);
        if (abortRef.current === ac) abortRef.current = null;
        promptRef.current?.focus();
      }
    },
    [settings, appendMessage, appendDelta, finalizeAssistant, updateActive, setBusyBoth],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const send = useCallback(
    async (textArg?: string) => {
      const promptEl = promptRef.current;
      if (!promptEl) return;
      const q = (textArg != null ? textArg : promptEl.value).trim();
      if (!q) return;

      // Local buddy on/off — no model wait; runs even while a reply is generating.
      // Accepts: /buddy on|off, --buddy on|off, plain "buddy on|off" (+ optional karo/kardo)
      const onOff = matchBuddyOnOff(q);
      if (onOff) {
        promptEl.value = "";
        autoGrow();
        window.sentinelBuddySetVisible?.(onOff === "on");
        // Status stays on the float — avoid chat clutter
        appendMessage(
          "assistant",
          onOff === "off"
            ? "Buddy hidden."
            : "Buddy on — linked to you + AERIS-64. Hatch progress only on the float (not in chat).",
        );
        return;
      }
      if (busyRef.current) {
        promptEl.placeholder = "⌛ waiting for reply…";
        window.setTimeout(() => {
          if (promptRef.current) promptRef.current.placeholder = PROMPT_PLACEHOLDER;
        }, 1600);
        return;
      }

      promptEl.value = "";
      autoGrow();

      // Egg incubation: one step per real user message (fire-and-forget)
      try {
        window.sentinelBuddyStep?.();
      } catch (_) {}

      const activeChat = storeRef.current.chats.find((c) => c.id === storeRef.current.currentId);

      if (editingId != null) {
        // Edited message: drop it + everything after, reset the server session,
        // then resend the new text as a fresh user turn.
        const targetId = editingId;
        setEditingId(null);
        forgetSession();
        updateActive((c) => {
          const kept = c.messages.filter((m) => m.id < targetId);
          return {
            ...c,
            sessionId: null,
            updatedAt: Date.now(),
            title: kept.length === 0 ? titleFrom(q) : c.title,
            messages: kept,
          };
        });
        await runPrompt(q, {
          appendUser: true,
          sessionId: null,
          threadId: activeChat?.id || "web-session",
        });
        return;
      }

      const sessionId = activeChat?.sessionId ?? null;
      const threadId = sessionId || activeChat?.id || "web-session";
      await runPrompt(q, { appendUser: true, sessionId, threadId });
    },
    [editingId, appendMessage, autoGrow, forgetSession, runPrompt, updateActive],
  );

  const regenerate = useCallback(async () => {
    if (busyRef.current) return;
    const msgs = active.messages;
    let targetIdx = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i]!.role === "assistant") {
        targetIdx = i;
        break;
      }
    }
    if (targetIdx < 0) return;
    const target = msgs[targetIdx]!;
    let userText: string | null = null;
    for (let i = targetIdx - 1; i >= 0; i--) {
      if (msgs[i]!.role === "user") {
        userText = msgs[i]!.text;
        break;
      }
    }
    if (userText == null) return;

    forgetSession();
    updateActive((c) => ({
      ...c,
      sessionId: null,
      updatedAt: Date.now(),
      messages: c.messages.filter((m) => m.id < target.id),
    }));
    await runPrompt(userText, {
      appendUser: false,
      sessionId: null,
      threadId: active.id,
    });
  }, [active, forgetSession, runPrompt, updateActive]);

  const startEdit = useCallback(
    (msg: ChatMessage) => {
      const el = promptRef.current;
      if (!el) return;
      el.value = msg.text;
      autoGrow();
      setEditingId(msg.id);
      el.focus();
    },
    [autoGrow],
  );

  const cancelEdit = useCallback(() => setEditingId(null), []);

  const lastAssistantIdx = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i]!.role === "assistant") return i;
    }
    return -1;
  }, [messages]);

  /* ---- account gate: no session yet ------------------------------------- */

  if (!session) {
    if (authStatus !== "checking" && authStatus && authStatus.accounts) {
      return <AuthGate status={authStatus} onAuthed={enterSession} />;
    }
    return (
      <div className="boot" id="boot">
        <img src="/logo.svg" alt="" width={44} height={44} />
        <p>Starting SENTINEL…</p>
      </div>
    );
  }

  const accountUser = session.kind === "user" ? session.user : null;

  return (
    <div className="shell">
      <Sidebar
        chats={visibleChats}
        currentId={store.currentId}
        open={sidebarOpen}
        onSelect={selectChat}
        onDelete={deleteChat}
        onNew={newChat}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="main-col">
        <Topbar
          health={health}
          menuOpen={sidebarOpen}
          user={accountUser}
          modeLabel={settings.effort === "fast" ? "Fast" : settings.effort === "think" ? "Think" : "AERIS-64"}
          onMenu={() => setSidebarOpen((v) => !v)}
          onNew={newChat}
          onLogout={handleLogout}
          onLogin={requestLogin}
          onMode={() => setModeOpen(true)}
        />
        <main id="chat" role="log" aria-live="polite" aria-relevant="additions" ref={chatRef}>
          <div className="thread" id="thread">
            {messages.length === 0 && !typing && (
              <Welcome onChip={(q) => send(q)} userName={accountUser?.name || accountUser?.email || null} />
            )}
            {messages.map((m, i) => (
              <Message
                key={m.id}
                msg={m}
                canEdit={!busy && m.role === "user"}
                canRegenerate={!busy && i === lastAssistantIdx}
                onEdit={startEdit}
                onRegenerate={regenerate}
              />
            ))}
            {typing && <TypingIndicator />}
          </div>
        </main>
        <Composer
          promptRef={promptRef}
          busy={busy}
          editing={editingId != null}
          maxTok={settings.maxTok}
          temp={settings.temp}
          autoLen={settings.autoLen}
          latencyText={latencyText}
          onSettings={updateSettings}
          onInput={autoGrow}
          onSubmit={() => send()}
          onStop={stop}
          onClear={clearChat}
          onCancelEdit={cancelEdit}
        />
      </div>
      <ModeSheet
        open={modeOpen}
        effort={settings.effort}
        context={settings.context}
        onClose={() => setModeOpen(false)}
        onEffort={(effort) => {
          const maxTok = effort === "fast" ? "48" : effort === "think" ? "256" : "96";
          updateSettings({ effort, maxTok });
        }}
        onContext={(context) => updateSettings({ context })}
      />
      <BuddyFloat />
    </div>
  );
}
