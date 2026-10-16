import { useEffect, useRef, useState } from "react";
import { renderInto } from "../lib/markdown";

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  meta?: string;
  /** true while tokens are still streaming into this message */
  streaming?: boolean;
}

function CopyButton({ text }: { text: string }) {
  const [label, setLabel] = useState("Copy");
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setLabel("Copied");
          window.setTimeout(() => setLabel("Copy"), 1200);
        } catch (_) {}
      }}
    >
      {label}
    </button>
  );
}

interface MessageProps {
  msg: ChatMessage;
  canEdit?: boolean;
  canRegenerate?: boolean;
  onEdit?: (msg: ChatMessage) => void;
  onRegenerate?: () => void;
}

export function Message({ msg, canEdit, canRegenerate, onEdit, onRegenerate }: MessageProps) {
  const bubbleRef = useRef<HTMLDivElement | null>(null);
  const isUser = msg.role === "user";
  const streaming = !!msg.streaming;

  useEffect(() => {
    if (!isUser && bubbleRef.current && !streaming) renderInto(bubbleRef.current, msg.text);
  }, [isUser, msg.text, streaming]);

  const showActions =
    !streaming && !!msg.text && !String(msg.text).startsWith("Error") && !String(msg.text).startsWith("Network error");

  return (
    <article
      className={"msg " + (isUser ? "user" : "assistant") + (streaming ? " streaming" : "")}
    >
      <div className="role">{isUser ? "You" : "SENTINEL"}</div>
      {/* Assistant bubbles are filled imperatively by renderInto() once complete
          (marked + DOMPurify + hljs); while streaming we render the raw text. */}
      <div className="bubble" ref={bubbleRef}>
        {isUser || streaming ? msg.text : null}
      </div>
      {showActions && (
        <div className="actions">
          <CopyButton text={msg.text} />
          {isUser && canEdit && onEdit && (
            <button
              type="button"
              className="edit"
              title="Edit and resend this message"
              onClick={() => onEdit(msg)}
            >
              ✎ Edit
            </button>
          )}
          {!isUser && canRegenerate && onRegenerate && (
            <button
              type="button"
              className="regen"
              title="Generate a new reply"
              onClick={onRegenerate}
            >
              ↻ Regenerate
            </button>
          )}
        </div>
      )}
      {msg.meta && <div className="meta">{msg.meta}</div>}
    </article>
  );
}

export function TypingIndicator() {
  return (
    <article id="typing" className="msg assistant">
      <div className="role">SENTINEL</div>
      <div className="bubble">
        <span className="typing">
          <i></i>
          <i></i>
          <i></i>
        </span>
      </div>
    </article>
  );
}
