export interface Health {
  state: "unknown" | "ok" | "err";
  label: string;
}

export function Topbar({ health, onNew }: { health: Health; onNew: () => void }) {
  const cls =
    "status" + (health.state === "ok" ? " ok" : health.state === "err" ? " err" : "");
  return (
    <header className="topbar" role="banner">
      <div className="brand">
        <img src="/logo.svg" alt="" width={28} height={28} />
        <div className="brand-text">
          <strong>SENTINEL</strong>
          <span>Think Smarter. Do More</span>
        </div>
      </div>
      <nav className="nav" aria-label="Primary">
        <a href="/">Chat</a>
        <a href="/product">Product</a>
      </nav>
      <div className="topbar-right">
        <div id="status" className={cls} title="Backend health">
          <span className="dot" aria-hidden="true"></span>
          <span id="statusText">{health.label}</span>
        </div>
        <button
          type="button"
          className="icon-btn"
          id="btnNew"
          title="New chat"
          aria-label="New chat"
          onClick={onNew}
        >
          ↻
        </button>
      </div>
    </header>
  );
}
