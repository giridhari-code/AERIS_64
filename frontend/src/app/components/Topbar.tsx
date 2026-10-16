import { useEffect, useRef, useState } from "react";
import type { AuthUser } from "../lib/auth";

export interface Health {
  state: "unknown" | "ok" | "err";
  label: string;
}

interface TopbarProps {
  health: Health;
  menuOpen: boolean;
  /** Signed-in account, or null for guests / undecided. */
  user: AuthUser | null;
  modeLabel?: string;
  onMenu: () => void;
  onNew: () => void;
  onLogout: () => void;
  onLogin: () => void;
  onMode?: () => void;
}

export function Topbar({ health, menuOpen, user, modeLabel, onMenu, onNew, onLogout, onLogin, onMode }: TopbarProps) {
  const [accountOpen, setAccountOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement | null>(null);

  /* Close the account menu on outside click / Escape. */
  useEffect(() => {
    if (!accountOpen) return;
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setAccountOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setAccountOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [accountOpen]);

  const cls =
    "status" + (health.state === "ok" ? " ok" : health.state === "err" ? " err" : "");
  const initial = user ? (user.name.trim()[0] || "?").toUpperCase() : "👤";

  return (
    <header className="topbar" role="banner">
      <button
        type="button"
        className="icon-btn"
        id="btnMenu"
        title="Chats"
        aria-label="Chats"
        aria-pressed={menuOpen}
        onClick={onMenu}
      >
        ☰
      </button>
      <div className="brand">
        <img src="/logo.svg" alt="" width={28} height={28} />
        <div className="brand-text">
          <strong>SENTINEL</strong>
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
        {onMode && (
          <button type="button" className="mode-pill" onClick={onMode} title="Model & effort">
            {modeLabel || "Fable"}
          </button>
        )}
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
        <div className="user-box" ref={boxRef}>
          <button
            type="button"
            className={"user-chip" + (user ? " signed-in" : "")}
            id="btnUser"
            title={user ? `${user.name} (${user.email})` : "Guest — log in or sign up"}
            aria-label="Account"
            aria-haspopup="menu"
            aria-expanded={accountOpen}
            onClick={() => setAccountOpen((v) => !v)}
          >
            {initial}
          </button>
          {accountOpen && (
            <div className="user-menu" id="userMenu" role="menu">
              <div className="user-menu-head">
                <strong>{user ? user.name : "Guest"}</strong>
                <span>{user ? user.email : "Chats stay on this device"}</span>
              </div>
              {user ? (
                <button
                  type="button"
                  role="menuitem"
                  id="btnLogout"
                  onClick={() => {
                    setAccountOpen(false);
                    onLogout();
                  }}
                >
                  Log out
                </button>
              ) : (
                <button
                  type="button"
                  role="menuitem"
                  id="btnLogin"
                  onClick={() => {
                    setAccountOpen(false);
                    onLogin();
                  }}
                >
                  Log in / Sign up
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
