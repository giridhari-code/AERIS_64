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
          <div className="composer-bar">
            <button
              type="button"
              className="composer-icon"
              title="New chat / clear"
              aria-label="Clear chat"
              onClick={p.onClear}
            >
              +
            </button>
            {p.onMode && (
              <button
                type="button"
                className="mode-chip"
                onClick={p.onMode}
                title="Model & effort"
              >
                {p.modeLabel || "Fable · recommend"}
              </button>
            )}
            <span className="composer-bar-spacer" />
            {p.latencyText ? <span className="lat-inline">{p.latencyText}</span> : null}
            {p.busy ? (
              <button
                type="button"
                className="composer-icon stop-icon"
                title="Stop"
                aria-label="Stop"
                onClick={p.onStop}
              >
                ■
              </button>
            ) : (
              <button type="submit" className="composer-icon send-icon" title="Send" aria-label="Send">
                ↑
              </button>
            )}
          </div>
        </form>
      </div>
    </footer>
  );
}
