/**
 * Product mode sheet — model lineup (Blink / Fable / Mythos / Oracle) + effort.
 */
import {
  EFFORTS,
  MODELS,
  type EffortId,
  type ModelId,
} from "../lib/models";

interface ModeSheetProps {
  open: boolean;
  model: ModelId;
  effort: EffortId;
  onClose: () => void;
  onModel: (m: ModelId) => void;
  onEffort: (e: EffortId) => void;
}

export function ModeSheet(p: ModeSheetProps) {
  if (!p.open) return null;
  return (
    <div className="mode-sheet-root" role="dialog" aria-modal="true" aria-label="Model and effort">
      <button type="button" className="mode-sheet-backdrop" aria-label="Close" onClick={p.onClose} />
      <div className="mode-sheet">
        <div className="mode-sheet-head">
          <div>
            <strong>Model</strong>
            <p className="mode-sheet-hint">SENTINEL lineup · one checkpoint, tuned decode budget</p>
          </div>
          <button type="button" className="icon-btn" onClick={p.onClose} aria-label="Close">
            ×
          </button>
        </div>

        <ul className="mode-list" role="listbox" aria-label="Models">
          {MODELS.map((m) => (
            <li key={m.id}>
              <button
                type="button"
                role="option"
                aria-selected={p.model === m.id}
                className={"mode-row" + (p.model === m.id ? " selected" : "")}
                onClick={() => p.onModel(m.id)}
              >
                <span className="mode-row-text">
                  <span className="mode-row-title">{m.name}</span>
                  <span className="mode-row-desc">{m.tagline}</span>
                </span>
                {p.model === m.id && <span className="mode-check" aria-hidden="true">✓</span>}
              </button>
            </li>
          ))}
        </ul>

        <div className="mode-divider" />
        <p className="mode-section-label">Effort</p>
        <div className="effort-grid" role="listbox" aria-label="Effort">
          {EFFORTS.map((e) => (
            <button
              key={e.id}
              type="button"
              role="option"
              aria-selected={p.effort === e.id}
              className={"effort-chip" + (p.effort === e.id ? " selected" : "")}
              title={e.blurb}
              onClick={() => p.onEffort(e.id)}
            >
              <span className="effort-label">{e.label}</span>
              <span className="effort-meta">{e.maxTokens} tok</span>
            </button>
          ))}
        </div>
        <p className="mode-foot">
          Higher effort = more tokens / slower on CPU. Fable + Recommend = product default.
        </p>
      </div>
    </div>
  );
}
