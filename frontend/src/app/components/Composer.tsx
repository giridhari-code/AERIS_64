import type { RefObject } from "react";

export interface SettingsPatch {
  maxTok?: string;
  temp?: string;
  autoLen?: boolean;
}

interface ComposerProps {
  promptRef: RefObject<HTMLTextAreaElement | null>;
  busy: boolean;
  editing: boolean;
  maxTok: string;
  temp: string;
  autoLen: boolean;
  latencyText: string;
  modeLabel?: string;
  onMode?: () => void;
  onSettings: (patch: SettingsPatch) => void;
  onInput: () => void;
  onSubmit: () => void;
  onStop: () => void;
  onClear: () => void;
  onCancelEdit: () => void;
}

export function Composer(p: ComposerProps) {
  return (
    <footer className="composer-wrap">
      <div className="composer-inner">
        {p.editing && (
          <div className="editing-banner" id="editingBanner">
            <span>Editing — send to replace from here</span>
            <button type="button" className="link" onClick={p.onCancelEdit}>
              Cancel
            </button>
          </div>
        )}
        <form
          id="form"
          className="composer"
          autoComplete="off"
          onSubmit={(e) => {
            e.preventDefault();
            p.onSubmit();
          }}
        >
          <textarea
            id="prompt"
            rows={1}
            placeholder="Message SENTINEL…"
            aria-label="Message"
            ref={p.promptRef}
            onInput={p.onInput}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                p.onSubmit();
              } else if (e.key === "Escape" && p.editing) {
                e.preventDefault();
                p.onCancelEdit();
              }
            }}
          />
          {p.busy ? (
            <button type="button" className="stop" title="Stop" aria-label="Stop" onClick={p.onStop}>
              ■
            </button>
          ) : (
            <button type="submit" className="send" title="Send" aria-label="Send">
              ↑
            </button>
          )}
        </form>
        <div className="composer-meta">
          <div className="composer-meta-left">
            {p.onMode && (
              <button type="button" className="mode-pill footer-mode" onClick={p.onMode} title="Model & effort">
                {p.modeLabel || "Fable"}
              </button>
            )}
            {p.latencyText ? <span className="lat">{p.latencyText}</span> : null}
          </div>
          <button type="button" className="link quiet" onClick={p.onClear}>
            Clear
          </button>
        </div>
      </div>
    </footer>
  );
}
