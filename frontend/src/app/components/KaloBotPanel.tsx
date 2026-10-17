import { useCallback, useEffect, useRef, useState } from "react";

interface Child {
  id: string;
  name: string;
  role: string;
}

interface KaloIdentity {
  name?: string;
  tagline?: string;
  engine?: string;
  model?: string;
  children?: Child[];
  error?: string;
}

interface KaloEvent {
  name: string;
  ok: boolean;
  detail: string;
  ms: number;
}

interface Turn {
  role: "user" | "assistant";
  text: string;
  route?: string;
  reason?: string;
  events?: KaloEvent[];
  latency_ms?: number;
}

export function KaloBotPanel({
  onBack,
  onOpenDataset,
  onOpenChat,
}: {
  onBack: () => void;
  onOpenDataset?: () => void;
  onOpenChat?: () => void;
}) {
  const [identity, setIdentity] = useState<KaloIdentity | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let c = false;
    (async () => {
      try {
        const r = await fetch("/v1/kalo");
        const j = (await r.json()) as KaloIdentity;
        if (!c) setIdentity(j);
      } catch (e) {
        if (!c) setErr(String((e as Error).message || e));
      }
    })();
    return () => {
      c = true;
    };
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [turns, busy]);

  const send = useCallback(async () => {
    const el = inputRef.current;
    const q = (el?.value || "").trim();
    if (!q || busy) return;
    if (el) el.value = "";
    setBusy(true);
    setErr("");
    setTurns((t) => [...t, { role: "user", text: q }]);
    try {
      const r = await fetch("/v1/kalo/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: q, max_new_tokens: 96, temperature: 0.4 }),
      });
      const j = await r.json();
      if (!r.ok || j.ok === false) {
        setTurns((t) => [
          ...t,
          { role: "assistant", text: j.error || "KALO run failed", route: "error" },
        ]);
      } else {
        setTurns((t) => [
          ...t,
          {
            role: "assistant",
            text: j.answer || "(empty)",
            route: j.route,
            reason: j.route_reason,
            events: j.events,
            latency_ms: j.latency_ms,
          },
        ]);
      }
    } catch (e) {
      setTurns((t) => [
        ...t,
        { role: "assistant", text: String((e as Error).message || e), route: "error" },
      ]);
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }, [busy]);

  const name = identity?.name || "KALO BOT";
  const tagline = identity?.tagline || "Think Smarter. Do More.";

  return (
    <div className="db-panel kalo-panel" role="region" aria-label="KALO BOT">
      <header className="db-head">
        <button type="button" className="db-back" onClick={onBack} aria-label="Back">
          ←
        </button>
        <div className="db-title">
          <strong>{name}</strong>
          <span>
            {tagline}
            {identity?.engine ? ` · ${identity.engine}` : ""}
            {identity?.model ? ` · ${identity.model}` : ""}
          </span>
        </div>
        <button type="button" className="db-refresh" onClick={onOpenChat}>
          Chat
        </button>
      </header>

      <div className="db-scroll kalo-scroll" ref={listRef}>
        <section className="db-card kalo-hero">
          <h3>{name}</h3>
          <p className="kalo-tag">{tagline}</p>
          <p className="db-meta">
            Mother agent — routes to Bhutu, Megha, Buddy, Dataset Builder, tools, then SENTINEL when needed.
          </p>
          {err && <p className="db-err">{err}</p>}
        </section>

        {!!identity?.children?.length && (
          <section className="db-card">
            <h3>Children</h3>
            <div className="kalo-grid">
              {identity.children.map((c) => (
                <div key={c.id} className="kalo-chip">
                  <strong>{c.name}</strong>
                  <span>{c.role}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="db-card kalo-thread">
          <h3>Session</h3>
          {turns.length === 0 && (
            <p className="db-meta">Try: “buddy status”, “dataset progress”, “weather in Delhi”, or a normal question.</p>
          )}
          {turns.map((m, i) => (
            <div key={i} className={"kalo-msg " + m.role}>
              <div className="kalo-msg-who">
                {m.role === "user" ? "You" : name}
                {m.route ? ` · ${m.route}` : ""}
                {m.latency_ms != null ? ` · ${Math.round(m.latency_ms)}ms` : ""}
              </div>
              <div className="kalo-msg-body">{m.text}</div>
              {!!m.events?.length && (
                <div className="kalo-events">
                  {m.events.map((e, j) => (
                    <span key={j} className={"kalo-ev " + (e.ok ? "ok" : "bad")}>
                      {e.name}
                      {e.ms ? ` ${Math.round(e.ms)}ms` : ""}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {busy && <p className="db-meta">KALO routing…</p>}
        </section>

        <section className="db-card">
          <div className="db-row">
            <button type="button" className="db-btn secondary" onClick={onOpenDataset}>
              Dataset Builder
            </button>
            <button type="button" className="db-btn secondary" onClick={onOpenChat}>
              Main Chat
            </button>
          </div>
        </section>
      </div>

      <div className="kalo-composer">
        <textarea
          ref={inputRef}
          className="db-input db-area"
          rows={2}
          placeholder="Message KALO BOT…"
          disabled={busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <button type="button" className="db-btn" disabled={busy} onClick={() => void send()}>
          Send
        </button>
      </div>
    </div>
  );
}
