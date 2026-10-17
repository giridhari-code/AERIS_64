import { useEffect, useState } from "react";

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
  repo?: string;
  preamble?: string;
  children?: Child[];
  error?: string;
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
  const [data, setData] = useState<KaloIdentity | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/v1/kalo");
        const j = (await r.json()) as KaloIdentity;
        if (!cancelled) setData(j);
      } catch (e) {
        if (!cancelled) setErr(String((e as Error).message || e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const name = data?.name || "KALO BOT";
  const tagline = data?.tagline || "Think Smarter. Do More.";

  return (
    <div className="db-panel kalo-panel" role="region" aria-label="KALO BOT">
      <header className="db-head">
        <button type="button" className="db-back" onClick={onBack} aria-label="Back">
          ←
        </button>
        <div className="db-title">
          <strong>{name}</strong>
          <span>{tagline}</span>
        </div>
        <button type="button" className="db-refresh" onClick={onOpenChat}>
          Chat
        </button>
      </header>

      <div className="db-scroll">
        <section className="db-card kalo-hero">
          <h3>{name}</h3>
          <p className="kalo-tag">{tagline}</p>
          <p className="db-meta">
            Product face of <strong>AERIS_64</strong>
            {data?.engine ? ` · engine ${data.engine}` : ""}
            {data?.model ? ` · model ${data.model}` : ""}
          </p>
          {data?.preamble && <p className="kalo-preamble">{data.preamble}</p>}
          {err && <p className="db-err">{err}</p>}
          {data?.error && <p className="db-err">{data.error}</p>}
        </section>

        <section className="db-card">
          <h3>Agents & tools</h3>
          <div className="kalo-grid">
            {(data?.children || []).map((c) => (
              <div key={c.id} className="kalo-chip">
                <strong>{c.name}</strong>
                <span>{c.role}</span>
              </div>
            ))}
            {!data?.children?.length && (
              <p className="db-meta">Loading identity… start backend for live list.</p>
            )}
          </div>
        </section>

        <section className="db-card db-actions">
          <h3>Quick open</h3>
          <div className="db-row">
            <button type="button" className="db-btn" onClick={onOpenChat}>
              Open Chat
            </button>
            <button type="button" className="db-btn secondary" onClick={onOpenDataset}>
              Dataset Builder
            </button>
          </div>
          <p className="db-meta" style={{ marginTop: 10 }}>
            Chat uses AERIS_64 backend · Buddy float is outside the thread · Dataset Builder is under Tools.
          </p>
        </section>
      </div>
    </div>
  );
}
