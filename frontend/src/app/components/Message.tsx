import { useEffect, useRef, useState } from "react";
import { renderInto } from "../lib/markdown";

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  meta?: string;
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

export function Message({ msg }: { msg: ChatMessage }) {
  const bubbleRef = useRef<HTMLDivElement | null>(null);
  const isUser = msg.role === "user";

  useEffect(() => {
    if (!isUser && bubbleRef.current) renderInto(bubbleRef.current, msg.text);
  }, [isUser, msg.text]);

  const showActions = !isUser && !!msg.text && !String(msg.text).startsWith("Error");

  return (
    <article className={"msg " + (isUser ? "user" : "assistant")}>
      <div className="role">{isUser ? "You" : "SENTINEL"}</div>
      {/* Assistant bubbles are filled imperatively by renderInto() (marked + DOMPurify + hljs) */}
      <div className="bubble" ref={bubbleRef}>
        {isUser ? msg.text : null}
      </div>
      {showActions && (
        <div className="actions">
          <CopyButton text={msg.text} />
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
