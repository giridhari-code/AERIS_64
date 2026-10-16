"""Buddy egg — Pokémon-style incubation before the companion reveals.

Progress model (chosen by the user):
  * every user chat message is one step   (``POST /v1/buddy/step``)
  * passive time is also one step per ``NEUROFIELD_BUDDY_EGG_PERIOD``
    seconds (default 300 = 5 min)

Steps required are deterministic per user id (6–10), so the same account
always works toward the same goal. "Hatched" is derived
(``progress >= required``): the companion itself stays deterministic
(``hatch_buddy``) — the egg only delays the reveal, so existing devices
keep the buddy they already had.

State lives in a small JSON file (``NEUROFIELD_BUDDY_DB``, default
``data/buddy/eggs.json``) with thread-safe, atomic writes. Pure stdlib.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PERIOD_S = 300.0
_DB_ENV = "NEUROFIELD_BUDDY_DB"
_PERIOD_ENV = "NEUROFIELD_BUDDY_EGG_PERIOD"


# --------------------------------------------------------------------------- art

_EGG_BODY = [
    "   .-~~~-. ",
    "  / o   o \\",
    " :  ~~~~~ :",
    "  `-.___.-'",
]
_EGG_BLINK = [
    "   .-~~~-. ",
    "  / -   - \\",
    " :  ~~~~~ :",
    "  `-.___.-'",
]


def _frame(rows: list[str], width: int = 11) -> str:
    return "\n".join(r.ljust(width)[:width] for r in rows)


def _shift(rows: list[str], dx: int) -> list[str]:
    if dx < 0:
        return [r[-dx:] for r in rows]
    return [" " * dx + r for r in rows]


def egg_frames() -> list[str]:
    """Gentle wobble cycle (rest → tilt left → blink → tilt right)."""
    return [
        _frame(_EGG_BODY),
        _frame(_shift(_EGG_BODY, -1)),
        _frame(_EGG_BLINK),
        _frame(_shift(_EGG_BODY, 1)),
    ]


def crack_frames() -> list[str]:
    """Hatch sequence: hairline crack → squint → split → pop."""
    return [
        _frame(["   .-~_~-. ", "  / o   o \\", " :  ~~~~~ :", "  `-.___.-'"]),
        _frame(["   .-~_~-. ", "  / -   - \\", " :  ~_~_~ :", "  `-.___.-'"]),
        _frame(["  .-~_~-.  ", " / -   - \\ ", ":  ~ _ ~ :", " `-._._.-' "]),
        _frame(["   .-~ ~-. ", "  / *   * \\", " :  ~   ~ :", "  `-.___.-'"]),
    ]


# ---------------------------------------------------------------------- model


def egg_period_s() -> float:
    """Seconds per passive incubation step (env override for demos/tests)."""
    raw = os.environ.get(_PERIOD_ENV) or os.environ.get("BUDDY_EGG_PERIOD") or ""
    try:
        v = float(str(raw).strip())
        return v if v > 0 else DEFAULT_PERIOD_S
    except (TypeError, ValueError):
        return DEFAULT_PERIOD_S


def required_steps(user_id: str) -> int:
    """Deterministic 6–10 steps for this user (same account, same goal)."""
    dig = hashlib.sha256(f"sentinel-buddy-egg-v1:{user_id}".encode()).hexdigest()
    return 6 + (int(dig[:2], 16) % 5)


def _public(state: dict[str, Any]) -> dict[str, Any]:
    """Trim an internal snapshot to the wire shape (art only while unhatched)."""
    out: dict[str, Any] = {
        "messages": int(state.get("messages") or 0),
        "progress": int(state.get("progress") or 0),
        "required": int(state.get("required") or 1),
        "hatched": bool(state.get("hatched")),
        "period_s": float(state.get("period_s") or DEFAULT_PERIOD_S),
        "next_step_in_s": int(state.get("next_step_in_s") or 0),
    }
    if not out["hatched"]:
        out["frames"] = egg_frames()
        out["crack_frames"] = crack_frames()
    return out


def egg_status_text(state: dict[str, Any]) -> str:
    """Chat-friendly egg status (markdown, ASCII art fenced)."""
    progress = int(state.get("progress") or 0)
    required = int(state.get("required") or 1)
    if state.get("hatched"):
        return f"Buddy egg — already hatched ({progress}/{required})."
    art = egg_frames()[0]
    mins = int((int(state.get("next_step_in_s") or 0)) // 60)
    nxt = f"next passive step in ~{mins} min" if mins >= 1 else "next passive step due any moment"
    return (
        f"Buddy egg — {progress}/{required} steps\n"
        f"```\n{art}\n```\n"
        f"(har chat message = 1 step · time bhi count karta hai — {nxt})"
    )


# ---------------------------------------------------------------------- store


class EggStore:
    """Tiny JSON-file store: ``{user_id: {created_at, messages}}``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None

    # -- internals ---------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        users: dict[str, Any] = {}
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("users"), dict):
                    users = raw["users"]
        except Exception as e:  # noqa: BLE001 - corrupt file must not break chat
            logger.warning("buddy egg store unreadable (%s) — starting fresh: %s", self.path, e)
        self._data = {"version": 1, "users": users}
        return self._data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _uid(user_id: str) -> str:
        return str(user_id or "").strip() or "anonymous"

    def _record(self, data: dict[str, Any], uid: str, now: float) -> dict[str, Any]:
        rec = data["users"].get(uid)
        if not isinstance(rec, dict):
            rec = {"created_at": now, "messages": 0}
            data["users"][uid] = rec
            self._save(data)
        return rec

    def _snapshot(self, data: dict[str, Any], uid: str, now: float) -> dict[str, Any]:
        rec = self._record(data, uid, now)
        messages = int(rec.get("messages") or 0)
        created = float(rec.get("created_at") or now)
        # Adopted guest eggs keep their original goal (buddy key pinned).
        required = required_steps(str(rec.get("buddy_key") or uid))
        period = egg_period_s()
        elapsed = max(0.0, now - created)
        time_steps = int(elapsed // period) if period > 0 else 0
        progress = min(messages + time_steps, required)
        hatched = progress >= required
        next_step_in_s = 0
        if not hatched and period > 0:
            next_step_in_s = math.ceil(period - (elapsed % period))
        return {
            "messages": messages,
            "progress": progress,
            "required": required,
            "hatched": hatched,
            "period_s": period,
            "next_step_in_s": next_step_in_s,
        }

    # -- public ------------------------------------------------------------

    def snapshot(self, user_id: str, now: float | None = None) -> dict[str, Any]:
        """Current egg state (creates the egg on first look — Pokémon-style)."""
        uid = self._uid(user_id)
        ts = time.time() if now is None else float(now)
        with self._lock:
            return _public(self._snapshot(self._load(), uid, ts))

    def add_message_steps(self, user_id: str, n: int = 1, now: float | None = None) -> dict[str, Any]:
        """One chat message = one step (``n`` clamped to a sane burst size)."""
        uid = self._uid(user_id)
        ts = time.time() if now is None else float(now)
        try:
            step = max(1, min(int(n), 25))
        except (TypeError, ValueError):
            step = 1
        with self._lock:
            data = self._load()
            rec = self._record(data, uid, ts)
            rec["messages"] = int(rec.get("messages") or 0) + step
            self._save(data)
            return _public(self._snapshot(data, uid, ts))

    def buddy_key(self, user_id: str) -> str | None:
        """Pinned companion identity for this user (guest egg adopted by an account)."""
        uid = self._uid(user_id)
        with self._lock:
            rec = self._load()["users"].get(uid)
        if isinstance(rec, dict) and rec.get("buddy_key"):
            return str(rec["buddy_key"])
        return None

    def adopt(self, src_user_id: str, dst_user_id: str, now: float | None = None) -> dict[str, Any]:
        """Move a guest/device egg into an account on first login.

        A *started* account egg (``messages > 0``) is never overwritten — the
        account keeps its own companion. An untouched account egg (created by
        merely opening the float) is replaced, so the login flow's race
        (float loads first, adopt lands right after) still carries the guest
        egg over. The moved record keeps ``buddy_key = src`` so the companion
        name/species stays exactly the same before and after login.
        """
        src, dst = self._uid(src_user_id), self._uid(dst_user_id)
        if src == dst:
            return {"adopted": False, "reason": "same identity"}
        ts = time.time() if now is None else float(now)
        with self._lock:
            data = self._load()
            dst_rec = data["users"].get(dst)
            if isinstance(dst_rec, dict) and int(dst_rec.get("messages") or 0) > 0:
                return {"adopted": False, "reason": "account already has a started egg"}
            rec = data["users"].get(src)
            if not isinstance(rec, dict):
                return {"adopted": False, "reason": "no guest egg to adopt"}
            data["users"].pop(src, None)
            rec["buddy_key"] = str(rec.get("buddy_key") or src)
            data["users"][dst] = rec
            self._save(data)
            return {
                "adopted": True,
                "buddy_key": rec["buddy_key"],
                "egg": _public(self._snapshot(data, dst, ts)),
            }
