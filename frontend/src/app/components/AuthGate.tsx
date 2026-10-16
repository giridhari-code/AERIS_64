import { useState } from "react";
import type { FormEvent } from "react";
import { login, signup, type AuthStatus, type Session } from "../lib/auth";

interface AuthGateProps {
  status: AuthStatus;
  onAuthed: (s: Session) => void;
}

/**
 * Full-screen login / signup card. Shown only when the backend reports
 * `accounts: true` and the device has no stored session. Guests can skip in
 * one click unless the server demands an account (`guest_ok: false`).
 */
export function AuthGate({ status, onAuthed }: AuthGateProps) {
  const [mode, setMode] = useState<"login" | "signup">(
    status.users && status.users > 0 ? "login" : "signup",
  );
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const switchMode = (m: "login" | "signup") => {
    setMode(m);
    setError("");
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setError("");
    if (mode === "signup" && !name.trim()) {
      setError("Please enter your name.");
      return;
    }
    if (!/^\S+@\S+\.\S+$/.test(email.trim())) {
      setError("Please enter a valid email address.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setBusy(true);
    try {
      const s =
        mode === "signup"
          ? await signup(name.trim(), email.trim(), password)
          : await login(email.trim(), password);
      onAuthed(s);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <div className="auth" id="auth">
      <div className="auth-card" id="authCard">
        <div className="auth-brand">
          <img src="/logo.svg" alt="" width={40} height={40} />
          <div className="auth-brand-text">
            <strong>SENTINEL</strong>
            <span>Think Smarter. Do More</span>
          </div>
        </div>

        <div className="auth-tabs" id="authTabs" role="tablist" aria-label="Account">
          <button
            type="button"
            role="tab"
            id="tabSignup"
            aria-selected={mode === "signup"}
            className={mode === "signup" ? "on" : ""}
            onClick={() => switchMode("signup")}
          >
            Sign up
          </button>
          <button
            type="button"
            role="tab"
            id="tabLogin"
            aria-selected={mode === "login"}
            className={mode === "login" ? "on" : ""}
            onClick={() => switchMode("login")}
          >
            Log in
          </button>
        </div>

        <form id="authForm" onSubmit={submit} noValidate>
          {mode === "signup" && (
            <label className="auth-field">
              <span>Name</span>
              <input
                id="authName"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                autoComplete="name"
                maxLength={40}
              />
            </label>
          )}
          <label className="auth-field">
            <span>Email</span>
            <input
              id="authEmail"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              inputMode="email"
              maxLength={200}
            />
          </label>
          <label className="auth-field">
            <span>Password</span>
            <input
              id="authPass"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "signup" ? "At least 8 characters" : "Your password"}
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              maxLength={200}
            />
          </label>

          {error && (
            <div className="auth-error" id="authError" role="alert">
              {error}
            </div>
          )}

          <button type="submit" className="auth-submit" id="authSubmit" disabled={busy}>
            {busy ? "Please wait…" : mode === "signup" ? "Create account" : "Log in"}
          </button>
        </form>

        {status.guest_ok ? (
          <>
            <div className="auth-or">
              <span>or</span>
            </div>
            <button
              type="button"
              className="auth-guest"
              id="btnGuest"
              onClick={() => onAuthed({ kind: "guest" })}
            >
              Continue as guest →
            </button>
            <p className="auth-note">Guest chats stay on this device only.</p>
          </>
        ) : (
          <p className="auth-note" id="authNote">
            This server requires an account to chat.
          </p>
        )}
      </div>
      <p className="auth-foot">
        Accounts live on your server (sqlite); passwords are scrypt-hashed and never stored in clear.
      </p>
    </div>
  );
}
