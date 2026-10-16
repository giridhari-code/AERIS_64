import { useCallback, useEffect, useRef, useState } from "react";
import { Topbar, type Health } from "./components/Topbar";
import { Welcome } from "./components/Welcome";
import { Message, TypingIndicator, type ChatMessage } from "./components/Message";
import { Composer } from "./components/Composer";
import { BuddyFloat } from "./components/BuddyFloat";
import { initBuddy } from "./lib/buddy-float";
import { matchBuddyOnOff } from "../buddy";

const PROMPT_PLACEHOLDER = "Message SENTINEL…";

interface Settings {
  maxTok: string;
  temp: string;
  autoLen: boolean;
}

function loadSettings(): Settings {
  try {
    const s = JSON.parse(localStorage.getItem("sentinel_ui") || "{}");
    return {
      maxTok: s.maxTok != null ? String(s.maxTok) : "64",
      temp: s.temp != null ? String(s.temp) : "0.4",
      autoLen: s.autoLen != null ? !!s.autoLen : true,
    };
  } catch (_) {
    return { maxTok: "64", temp: "0.4", autoLen: true };
  }
}

function fmtMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "";
  if (ms < 1000) return Math.round(ms) + " ms";
  return (ms / 1000).toFixed(1) + " s";
}

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [typing, setTyping] = useState(false);
  const [busy, setBusy] = useState(false);
  const [latencyText, setLatencyText] = useState("");
  const [health, setHealth] = useState<Health>({ state: "unknown", label: "…" });
  const [settings, setSettings] = useState<Settings>(loadSettings);

  const promptRef = useRef<HTMLTextAreaElement | null>(null);
  const chatRef = useRef<HTMLElement | null>(null);
  const idRef = useRef(0);
  const busyRef = useRef(false);
  const sessionRef = useRef<string | null>(localStorage.getItem("sentinel_session") || null);
  const epochRef = useRef(0);

  const setBusyBoth = useCallback((v: boolean) => {
    busyRef.current = v;
    setBusy(v);
  }, []);

  const appendMessage = useCallback((role: ChatMessage["role"], text: string, meta?: string) => {
    setMessages((prev) => [...prev, { id: ++idRef.current, role, text, meta }]);
  }, []);

  const autoGrow = useCallback(() => {
    const el = promptRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, []);

  /* Buddy float (imperative controller — same DOM ids/behavior as the legacy UI) */
  useEffect(() => {
    initBuddy();
  }, []);

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
          JSON.stringify({ maxTok: next.maxTok, temp: next.temp, autoLen: next.autoLen }),
        );
      } catch (_) {}
      return next;
    });
  }, []);

  const resetSession = useCallback(async () => {
    epochRef.current += 1; // invalidate any in-flight reply
    try {
      if (sessionRef.current) {
        await fetch("/v1/reset?session_id=" + encodeURIComponent(sessionRef.current), { method: "POST" });
      }
    } catch (_) {}
    sessionRef.current = null;
    localStorage.removeItem("sentinel_session");
    setMessages([]);
    setBusyBoth(false);
    setTyping(false);
    setLatencyText("");
  }, [setBusyBoth]);

  const send = useCallback(
    async (text?: string) => {
      const promptEl = promptRef.current;
      if (!promptEl) return;
      const q = (text != null ? text : promptEl.value).trim();
      if (!q) return;

      // Local buddy on/off — no model wait; runs even while a reply is generating.
      // Accepts: /buddy on|off, --buddy on|off, plain "buddy on|off" (+ optional karo/kardo)
      const onOff = matchBuddyOnOff(q);
      if (onOff) {
        promptEl.value = "";
        autoGrow();
        window.sentinelBuddySetVisible?.(onOff === "on");
        appendMessage("assistant", onOff === "off" ? "Buddy off." : "Buddy on.");
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
      appendMessage("user", q);
      setBusyBoth(true);
      setTyping(true);
      try {
        window.sentinelBuddyReact?.(q, "user");
      } catch (_) {}

      const maxTok = parseInt(settings.maxTok, 10) || 64;
      const temp = parseFloat(settings.temp);
      const auto = settings.autoLen;

      const epoch = epochRef.current;
      try {
        const r = await fetch("/v1/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: q,
            max_new_tokens: maxTok,
            temperature: Number.isFinite(temp) ? temp : 0.4,
            auto_length: auto,
            session_id: sessionRef.current || null,
          }),
        });
        if (epoch !== epochRef.current) return; // session reset while waiting — drop reply
        setTyping(false);
        if (!r.ok) {
          const err = await r.json().catch(() => ({}) as Record<string, unknown>);
          if (epoch !== epochRef.current) return;
          let detail = err.detail as unknown;
          if (Array.isArray(detail)) {
            detail = detail.map((d) => (d as { msg?: string }).msg || JSON.stringify(d)).join("; ");
          } else if (detail && typeof detail === "object") {
            detail = JSON.stringify(detail);
          }
          appendMessage("assistant", "Error: " + (detail || r.statusText || r.status));
        } else {
          const j = await r.json();
          if (epoch !== epochRef.current) return;
          sessionRef.current = j.session_id || sessionRef.current;
          if (sessionRef.current) localStorage.setItem("sentinel_session", sessionRef.current);
          const lat = j.latency_ms != null ? fmtMs(j.latency_ms) : "";
          const reply = (j.text || "").trim() || "(empty reply — train with more SFT data)";
          appendMessage("assistant", reply, lat ? "latency " + lat : "");
          try {
            window.sentinelBuddyReact?.(reply, "assistant");
          } catch (_) {}
          const bmeta = j.meta_rag || {};
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
        }
      } catch (e) {
        if (epoch !== epochRef.current) return;
        setTyping(false);
        appendMessage("assistant", "Network error: " + (e as Error).message);
      }
      setBusyBoth(false);
      promptRef.current?.focus();
    },
    [settings, appendMessage, autoGrow, setBusyBoth],
  );

  return (
    <>
      <Topbar health={health} onNew={resetSession} />
      <main id="chat" role="log" aria-live="polite" aria-relevant="additions" ref={chatRef}>
        <div className="thread" id="thread">
          {messages.length === 0 && !typing && <Welcome onChip={(q) => send(q)} />}
          {messages.map((m) => (
            <Message key={m.id} msg={m} />
          ))}
          {typing && <TypingIndicator />}
        </div>
      </main>
      <Composer
        promptRef={promptRef}
        busy={busy}
        maxTok={settings.maxTok}
        temp={settings.temp}
        autoLen={settings.autoLen}
        latencyText={latencyText}
        onSettings={updateSettings}
        onInput={autoGrow}
        onSubmit={() => send()}
        onClear={resetSession}
      />
      <BuddyFloat />
    </>
  );
}
