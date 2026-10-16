"""Local user accounts for the web client: signup / login / bearer sessions.

Deliberately dependency-free (stdlib only):

* **Passwords** — ``hashlib.scrypt`` with a per-user random salt; verification
  is a constant-time compare. A dummy verify runs for unknown emails so login
  timing does not reveal whether an account exists.
* **Sessions** — opaque random bearer tokens; only the SHA-256 of a token is
  stored, so a leaked database cannot be replayed. Tokens expire
  (``NEUROFIELD_AUTH_TTL_DAYS``, default 30 days) and logout deletes the row.
* **Storage** — SQLite (``NEUROFIELD_AUTH_DB``, default
  ``data/auth/accounts.db``; the repo gitignores that directory).

This layer gates the *web client* (nicer UX, per-user chat identity). Machine
auth on the model API still uses ``NEUROFIELD_API_KEYS``. With
``NEUROFIELD_AUTH_REQUIRED=1`` the chat endpoints additionally demand a valid
account token or API key (see ``require_chat_access`` in ``server.py``).
Set ``NEUROFIELD_AUTH=0`` to disable accounts entirely (old behavior).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path

_EMAIL_RE = re.compile(r"^[^@\s]{1,120}@[^@\s]{1,64}\.[^@\s]{1,64}$")

MIN_PASSWORD = 8
MAX_PASSWORD = 200
MIN_NAME = 1
MAX_NAME = 40

# scrypt work factors (16 MiB — fine on a phone, slow enough to matter).
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_TOKEN_BYTES = 32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    pw_salt    BLOB NOT NULL,
    pw_hash    BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id);
"""


class AccountError(Exception):
    """Validation / auth failure with an HTTP-ish status attached."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32
    )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _ttl_seconds() -> float:
    raw = os.environ.get("NEUROFIELD_AUTH_TTL_DAYS", "30")
    try:
        days = max(1e-5, float(raw))  # floor ≈ 0.86 s (lets tests exercise expiry)
    except ValueError:
        days = 30.0
    return days * 86400.0


class AccountStore:
    """SQLite-backed user + session store. One short-lived connection per call
    (auth traffic is tiny), so no cross-thread state to guard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _clean_name(name: str) -> str:
        clean = " ".join((name or "").split())
        if not (MIN_NAME <= len(clean) <= MAX_NAME):
            raise AccountError(422, f"name must be {MIN_NAME}-{MAX_NAME} characters")
        return clean

    @staticmethod
    def _clean_email(email: str) -> str:
        clean = (email or "").strip().lower()
        if not _EMAIL_RE.match(clean):
            raise AccountError(422, "enter a valid email address")
        return clean

    @staticmethod
    def _check_password(password: str) -> str:
        if not (MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD):
            raise AccountError(422, f"password must be at least {MIN_PASSWORD} characters")
        return password

    # -- users --------------------------------------------------------------

    def signup(self, name: str, email: str, password: str) -> tuple[dict, str]:
        """Create an account (and log it in). Returns ``(user, token)``."""
        name = self._clean_name(name)
        email = self._clean_email(email)
        password = self._check_password(password)

        salt = secrets.token_bytes(16)
        pw_hash = _scrypt(password, salt)
        uid = secrets.token_hex(8)
        now = time.time()
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT INTO users (id, name, email, pw_salt, pw_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (uid, name, email, salt, pw_hash, now),
                )
        except sqlite3.IntegrityError:
            raise AccountError(409, "email already registered") from None
        user = {"id": uid, "name": name, "email": email, "created_at": now}
        return user, self._issue_token(uid)

    def login(self, email: str, password: str) -> tuple[dict, str]:
        """Verify credentials. Returns ``(user, token)``; 401 on any mismatch."""
        email = (email or "").strip().lower()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, name, email, pw_salt, pw_hash, created_at FROM users "
                "WHERE email = ? COLLATE NOCASE",
                (email,),
            ).fetchone()
            if row is None:
                # Equalize timing with the found-user path.
                _scrypt(password or "", b"dummy-salt-16byt")
                raise AccountError(401, "invalid email or password")
            if not hmac.compare_digest(_scrypt(password or "", row["pw_salt"]), row["pw_hash"]):
                raise AccountError(401, "invalid email or password")
            user = {
                "id": row["id"],
                "name": row["name"],
                "email": row["email"],
                "created_at": row["created_at"],
            }
        return user, self._issue_token(user["id"])

    # -- sessions -----------------------------------------------------------

    def _issue_token(self, user_id: str) -> str:
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        now = time.time()
        with closing(self._connect()) as conn, conn:
            # opportunistic cleanup: drop expired sessions
            conn.execute("DELETE FROM tokens WHERE expires_at < ?", (now,))
            conn.execute(
                "INSERT INTO tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (_token_hash(token), user_id, now, now + _ttl_seconds()),
            )
        return token

    def verify(self, token: str | None) -> dict | None:
        """Return the user dict for a live token, else None."""
        if not token:
            return None
        now = time.time()
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT u.id, u.name, u.email, u.created_at, t.expires_at "
                "FROM tokens t JOIN users u ON u.id = t.user_id WHERE t.token_hash = ?",
                (_token_hash(token),),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] < now:
                conn.execute("DELETE FROM tokens WHERE token_hash = ?", (_token_hash(token),))
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "email": row["email"],
                "created_at": row["created_at"],
            }

    def logout(self, token: str | None) -> bool:
        if not token:
            return False
        with closing(self._connect()) as conn, conn:
            cur = conn.execute("DELETE FROM tokens WHERE token_hash = ?", (_token_hash(token),))
        return cur.rowcount > 0

    def user_count(self) -> int:
        with closing(self._connect()) as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(n)
