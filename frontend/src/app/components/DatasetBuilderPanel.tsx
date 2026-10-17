import { useCallback, useEffect, useState } from "react";

interface CatRow {
  name: string; target_tokens: number; current_tokens: number; doc_count: number;
  remaining?: number; progress_pct?: number;
}
interface DbState {
  ok: boolean; report: string; documents: number; tokens: number;
  categories: CatRow[]; needs_review: number; error?: string;
}
const CATS = ["general_language","science_factual","mathematics","reasoning_logic","hinglish"];

export function DatasetBuilderPanel({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<DbState | null>(null);
  const [busy, setBusy] = useState(false);
  const [paths, setPaths] = useState("");
  const [urls, setUrls] = useState("");
  const [text, setText] = useState("");
  const [category, setCategory] = useState("general_language");
  const [license, setLicense] = useState("unknown");
  const [fmt, setFmt] = useState("jsonl");
  const [msg, setMsg] = useState("");

  const run = useCallback(async (action: string, extra: Record<string, unknown> = {}) => {
    setBusy(true); setMsg("");
    try {
      const r = await fetch("/v1/tools/dataset_builder", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, category, license_note: license, export_format: fmt, ...extra }),
      });
      const j = (await r.json()) as DbState;
      setData(j);
      if (!r.ok || j.error) setMsg(j.error || "Request failed");
    } catch (e) { setMsg(String((e as Error).message || e)); }
    finally { setBusy(false); }
  }, [category, license, fmt]);

  useEffect(() => { void run("progress"); }, [run]);

  const cats = data?.categories || [];
  return (
    <div className="db-panel" role="region" aria-label="Dataset Builder">
      <header className="db-head">
        <button type="button" className="db-back" onClick={onBack} aria-label="Back to chat">←</button>
        <div className="db-title"><strong>Dataset Builder</strong><span>AERIS_64 · 1M token pack</span></div>
        <button type="button" className="db-refresh" disabled={busy} onClick={() => run("progress")}>Refresh</button>
      </header>
      <div className="db-scroll">
        <section className="db-card">
          <h3>Progress</h3>
          <p className="db-meta">docs {data?.documents ?? "—"} · tokens ≈ {data?.tokens ?? "—"} · review {data?.needs_review ?? "—"}</p>
          <div className="db-bars">
            {(cats.length ? cats : CATS.map((n) => ({ name: n, target_tokens: 1, current_tokens: 0, doc_count: 0, progress_pct: 0 }))).map((c) => {
              const pct = c.progress_pct ?? 0;
              return (
                <div key={c.name} className="db-bar-row">
                  <div className="db-bar-label"><span>{c.name.replace(/_/g, " ")}</span>
                    <span>{(c.current_tokens || 0).toLocaleString()} / {(c.target_tokens || 0).toLocaleString()}</span></div>
                  <div className="db-bar-track"><div className="db-bar-fill" style={{ width: Math.min(100, pct) + "%" }} /></div>
                </div>
              );
            })}
          </div>
        </section>
        <section className="db-card">
          <h3>Collect</h3>
          <label className="db-label">Category</label>
          <select className="db-input" value={category} onChange={(e) => setCategory(e.target.value)}>
            {CATS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <label className="db-label">License note</label>
          <input className="db-input" value={license} onChange={(e) => setLicense(e.target.value)} placeholder="public_domain / unknown" />
          <label className="db-label">File paths (one per line)</label>
          <textarea className="db-input db-area" rows={3} value={paths} onChange={(e) => setPaths(e.target.value)} placeholder="/data/essay.txt" />
          <button type="button" className="db-btn" disabled={busy} onClick={() => {
            const list = paths.split("\n").map((s) => s.trim()).filter(Boolean);
            if (!list.length) { setMsg("Add at least one file path"); return; }
            void run("collect_paths", { paths: list });
          }}>Add files</button>
          <label className="db-label">URLs (one per line)</label>
          <textarea className="db-input db-area" rows={3} value={urls} onChange={(e) => setUrls(e.target.value)} placeholder="https://…" />
          <button type="button" className="db-btn" disabled={busy} onClick={() => {
            const list = urls.split("\n").map((s) => s.trim()).filter(Boolean);
            if (!list.length) { setMsg("Add at least one URL"); return; }
            void run("collect_urls", { urls: list });
          }}>Fetch URLs</button>
          <label className="db-label">Paste text</label>
          <textarea className="db-input db-area" rows={4} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste article…" />
          <button type="button" className="db-btn" disabled={busy} onClick={() => {
            if (!text.trim()) { setMsg("Paste some text first"); return; }
            void run("add_text", { text });
          }}>Ingest text</button>
        </section>
        <section className="db-card">
          <h3>Balance & export</h3>
          <div className="db-row">
            <button type="button" className="db-btn secondary" disabled={busy} onClick={() => run("balance")}>Balance</button>
            <select className="db-input db-select" value={fmt} onChange={(e) => setFmt(e.target.value)}>
              <option value="jsonl">jsonl</option><option value="txt">txt</option><option value="csv">csv</option>
            </select>
            <button type="button" className="db-btn" disabled={busy} onClick={() => run("export")}>Export</button>
            <button type="button" className="db-btn secondary" disabled={busy} onClick={() => run("export_train")}>Training JSONL</button>
            <button type="button" className="db-btn secondary" disabled={busy} onClick={() => run("imbalance")}>Imbalance</button>
          </div>
        </section>
        {(msg || data?.report) && (
          <section className="db-card">
            <h3>Report</h3>
            {msg && <p className="db-err">{msg}</p>}
            <pre className="db-report">{data?.report || ""}</pre>
          </section>
        )}
      </div>
    </div>
  );
}
