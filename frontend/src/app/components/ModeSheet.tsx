/**
 * Friendly mode sheet — model effort + context length (Kimi-style clarity, SENTINEL branding).
 */
export type Effort = "fast" | "standard" | "think";
export type ContextMode = "standard" | "long";

interface ModeSheetProps {
  open: boolean;
  effort: Effort;
  context: ContextMode;
  onClose: () => void;
  onEffort: (e: Effort) => void;
  onContext: (c: ContextMode) => void;
}

const EFFORTS: Array<{ id: Effort; title: string; desc: string }> = [
  { id: "fast", title: "Fast", desc: "Quick replies, lower tokens" },
  { id: "standard", title: "AERIS-64", desc: "Balanced chat & tools" },
  { id: "think", title: "Think", desc: "More tokens, slower, careful" },
];

export function ModeSheet(p: ModeSheetProps) {
  if (!p.open) return null;
  return (
    <div className="mode-sheet-root" role="dialog" aria-label="Chat mode">
      <button type="button" className="mode-sheet-backdrop" aria-label="Close" onClick={p.onClose} />
      <div className="mode-sheet">
        <div className="mode-sheet-head">
          <strong>Mode</strong>
          <button type="button" className="icon-btn" onClick={p.onClose} aria-label="Close">
            ×
          </button>
        </div>
        <p className="mode-sheet-hint">Pick how hard SENTINEL works on this device.</p>
        <ul className="mode-list">
          {EFFORTS.map((e) => (
            <li key={e.id}>
              <button
                type="button"
                className={"mode-row" + (p.effort === e.id ? " selected" : "")}
                onClick={() => p.onEffort(e.id)}
              >
                <span className="mode-row-text">
                  <span className="mode-row-title">{e.title}</span>
                  <span className="mode-row-desc">{e.desc}</span>
                </span>
                {p.effort === e.id && <span className="mode-check">✓</span>}
              </button>
            </li>
          ))}
        </ul>
        <div className="mode-divider" />
        <p className="mode-section-label">Context length</p>
        <ul className="mode-list">
          <li>
            <button
              type="button"
              className={"mode-row" + (p.context === "standard" ? " selected" : "")}
              onClick={() => p.onContext("standard")}
            >
              <span className="mode-row-text">
                <span className="mode-row-title">Standard</span>
                <span className="mode-row-desc">Normal chat window</span>
              </span>
              {p.context === "standard" && <span className="mode-check">✓</span>}
            </button>
          </li>
          <li>
            <button
              type="button"
              className={"mode-row" + (p.context === "long" ? " selected" : "")}
              onClick={() => p.onContext("long")}
            >
              <span className="mode-row-text">
                <span className="mode-row-title">Long</span>
                <span className="mode-row-desc">More history in context (uses more memory)</span>
              </span>
              {p.context === "long" && <span className="mode-check">✓</span>}
            </button>
          </li>
        </ul>
      </div>
    </div>
  );
}
