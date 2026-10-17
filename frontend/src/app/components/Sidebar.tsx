import { relTime, type StoredChat } from "../lib/history";

export type SidebarPanel = "chat" | "dataset" | "kalo";

interface SidebarProps {
  chats: StoredChat[];
  currentId: string;
  open: boolean;
  activePanel?: SidebarPanel;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  onClose: () => void;
  onOpenDataset?: () => void;
  onOpenChat?: () => void;
  onOpenKalo?: () => void;
}

export function Sidebar(p: SidebarProps) {
  const panel = p.activePanel || "chat";
  return (
    <>
      <div
        className={"sidebar-backdrop" + (p.open ? " show" : "")}
        aria-hidden="true"
        onClick={p.onClose}
      ></div>
      <aside className={"sidebar" + (p.open ? " open" : "")} id="sidebar" aria-label="Navigation">
        <div className="sidebar-head">
          <button type="button" className="new-chat" id="btnNewSide" onClick={p.onNew}>
            ＋ New chat
          </button>
          <button
            type="button"
            className="icon-btn sidebar-close"
            id="btnSideClose"
            title="Close"
            aria-label="Close menu"
            onClick={p.onClose}
          >
            ×
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-label">Product</div>
          <button
            type="button"
            className={"side-tool" + (panel === "kalo" ? " active" : "")}
            onClick={() => {
              p.onOpenKalo?.();
              p.onClose();
            }}
          >
            <span className="side-tool-ico" aria-hidden="true">
              ◎
            </span>
            <span className="side-tool-text">
              <strong>KALO BOT</strong>
              <small>Think Smarter. Do More.</small>
            </span>
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-label">Tools</div>
          <button
            type="button"
            className={"side-tool" + (panel === "dataset" ? " active" : "")}
            onClick={() => {
              p.onOpenDataset?.();
              p.onClose();
            }}
          >
            <span className="side-tool-ico" aria-hidden="true">
              📦
            </span>
            <span className="side-tool-text">
              <strong>Dataset Builder</strong>
              <small>Collect · clean · 1M tokens</small>
            </span>
          </button>
          <button
            type="button"
            className={"side-tool" + (panel === "chat" ? " active" : "")}
            onClick={() => {
              p.onOpenChat?.();
              p.onClose();
            }}
          >
            <span className="side-tool-ico" aria-hidden="true">
              💬
            </span>
            <span className="side-tool-text">
              <strong>Chat</strong>
              <small>AERIS_64 assistant</small>
            </span>
          </button>
        </div>

        <div className="sidebar-section-label">Conversations</div>
        <nav className="chat-list" id="chatList" aria-label="Your chats">
          {p.chats.map((c) => (
            <div
              key={c.id}
              className={"chat-item" + (c.id === p.currentId && panel === "chat" ? " active" : "")}
            >
              <button
                type="button"
                className="chat-open"
                id={"chat-" + c.id}
                title={c.title}
                onClick={() => {
                  p.onOpenChat?.();
                  p.onSelect(c.id);
                }}
              >
                <span className="chat-title">{c.title}</span>
                <span className="chat-sub">
                  {relTime(c.updatedAt)}
                  {c.messages.length ? " · " + c.messages.length + " msg" : ""}
                </span>
              </button>
              <button
                type="button"
                className="chat-del"
                title="Delete chat"
                aria-label={"Delete chat: " + c.title}
                onClick={() => p.onDelete(c.id)}
              >
                ×
              </button>
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <a href="/">Chat</a> · KALO BOT · Dataset Builder
        </div>
      </aside>
    </>
  );
}
