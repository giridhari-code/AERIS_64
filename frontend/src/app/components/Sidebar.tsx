import { relTime, type StoredChat } from "../lib/history";

interface SidebarProps {
  chats: StoredChat[];
  currentId: string;
  open: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  onClose: () => void;
}

export function Sidebar(p: SidebarProps) {
  return (
    <>
      <div
        className={"sidebar-backdrop" + (p.open ? " show" : "")}
        aria-hidden="true"
        onClick={p.onClose}
      ></div>
      <aside className={"sidebar" + (p.open ? " open" : "")} id="sidebar" aria-label="Chat history">
        <div className="sidebar-head">
          <button type="button" className="new-chat" id="btnNewSide" onClick={p.onNew}>
            ＋ New chat
          </button>
          <button
            type="button"
            className="icon-btn sidebar-close"
            id="btnSideClose"
            title="Close"
            aria-label="Close chat history"
            onClick={p.onClose}
          >
            ×
          </button>
        </div>
        <nav className="chat-list" id="chatList" aria-label="Your chats">
          {p.chats.map((c) => (
            <div key={c.id} className={"chat-item" + (c.id === p.currentId ? " active" : "")}>
              <button
                type="button"
                className="chat-open"
                id={"chat-" + c.id}
                title={c.title}
                onClick={() => p.onSelect(c.id)}
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
          <a href="/">Chat</a> · <a href="/product">Product</a> · local model — quality depends
          on your checkpoint
        </div>
      </aside>
    </>
  );
}
