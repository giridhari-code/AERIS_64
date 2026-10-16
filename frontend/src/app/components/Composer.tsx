import type { RefObject } from "react";

export interface SettingsPatch {
  maxTok?: string;
  temp?: string;
  autoLen?: boolean;
}

interface ComposerProps {
  promptRef: RefObject<HTMLTextAreaElement | null>;
  busy: boolean;
  maxTok: string;
  temp: string;
  autoLen: boolean;
  latencyText: string;
  onSettings: (patch: SettingsPatch) => void;
  onInput: () => void;
  onSubmit: () => void;
  onClear: () => void;
}

export function Composer(p: ComposerProps) {
  return (
    <footer className="composer-wrap">
      <div className="composer-inner">
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
              }
            }}
          ></textarea>
          <button
            type="submit"
            className="send"
            id="send"
            title="Send"
            aria-label="Send"
            style={p.busy ? { opacity: 0.55 } : undefined}
          >
            ↑
          </button>
        </form>
        <div className="toolbar">
          <label title="Max new tokens">
            max
            <input
              type="number"
              id="maxTok"
              min={8}
              max={2048}
              value={p.maxTok}
              onChange={(e) => p.onSettings({ maxTok: e.target.value })}
            />
          </label>
          <label title="Temperature">
            temp
            <input
              type="number"
              id="temp"
              min={0}
              max={2}
              step={0.1}
              value={p.temp}
              onChange={(e) => p.onSettings({ temp: e.target.value })}
            />
          </label>
          <label title="Server auto length">
            <input
              type="checkbox"
              id="autoLen"
              checked={p.autoLen}
              onChange={(e) => p.onSettings({ autoLen: e.target.checked })}
            />
            auto
          </label>
          <span className="spacer"></span>
          <span id="latency">{p.latencyText}</span>
          <button type="button" className="link" id="btnClear" onClick={p.onClear}>
            Clear
          </button>
        </div>
        <p className="disclaimer">May produce inaccurate or nonsensical text. Verify important facts.</p>
      </div>
    </footer>
  );
}
