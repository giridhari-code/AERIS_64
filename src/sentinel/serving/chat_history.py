"""Multi-turn text context per session using chat special tokens when available."""

from __future__ import annotations

import os
from typing import Optional

from sentinel.tokenizer.specials import ASSISTANT, END, SYSTEM, USER

_HISTORY: dict[str, list[tuple[str, str]]] = {}
_MAX_TURNS = 6
_USE_SPECIALS = True  # set False to force legacy "User:" / "Assistant:" text
_SYSTEM_TEXT: str = ""


def set_use_specials(flag: bool) -> None:
    global _USE_SPECIALS
    _USE_SPECIALS = flag


def set_system_prompt(text: str) -> None:
    """Set global system prompt (prepended on every build_prompt)."""
    global _SYSTEM_TEXT
    _SYSTEM_TEXT = (text or "").strip()


def get_system_prompt() -> str:
    return _SYSTEM_TEXT


def default_system_prompt() -> str:
    """Company default — tools + behaviour (train data should mirror this)."""
    return (
        "You are SENTINEL, a helpful local research assistant. "
        "Be direct and honest about limits. "
        "For live facts (time, weather, news, math) the server may inject tool results — "
        "use those facts; do not invent numbers. "
        "If you need a tool, you may emit a single JSON line: "
        '{"tool": "time"|"calculator"|"web_search"|"weather"|"wikipedia", "query": "..."}. '
        "Otherwise answer in plain text."
    )


def reset_history(session_id: str | None = None) -> None:
    if session_id is None:
        _HISTORY.clear()
    else:
        _HISTORY.pop(session_id, None)


def restore_turns(session_id: str, turns: list[tuple[str, str]]) -> None:
    """Rebuild the in-process window from the persistent transcript (restart / TTL eviction)."""
    if _HISTORY.get(session_id):
        return
    h = _HISTORY.setdefault(session_id, [])
    for role, text in turns[-_MAX_TURNS * 2 :]:
        t = (text or "").strip()
        if t:
            h.append((role, t[:500]))


def build_user_part(user_prompt: str) -> str:
    """Just the new user turn (what build_prompt appends after the history)."""
    p = user_prompt.strip()
    return f"{USER}{p}{END}{ASSISTANT}" if _USE_SPECIALS else f"User: {p}
Assistant:"


def history_len(session_id: str) -> int:
    return len(_HISTORY.get(session_id, []))


def append_turn(session_id: str, role: str, text: str) -> None:
    t = (text or "").strip()
    if not t:
        return
    h = _HISTORY.setdefault(session_id, [])
    h.append((role, t[:500]))
    if len(h) > _MAX_TURNS * 2:
        del h[: len(h) - _MAX_TURNS * 2]


def build_prompt(
    session_id: str,
    user_prompt: str,
    system: Optional[str] = None,
) -> str:
    """
    Full chat context: [system] + history + user + assistant prefix.

    system=None → use global _SYSTEM_TEXT (or empty).
    """
    sys_text = (system if system is not None else _SYSTEM_TEXT) or ""
    parts: list[str] = []

    if _USE_SPECIALS:
        if sys_text:
            parts.append(f"{SYSTEM}{sys_text}{END}")
        for role, text in _HISTORY.get(session_id, []):
            tag = USER if role == "user" else ASSISTANT
            parts.append(f"{tag}{text}{END}")
        parts.append(f"{USER}{user_prompt.strip()}{END}")
        parts.append(ASSISTANT)
        return "".join(parts)

    # Legacy text format
    if sys_text:
        parts.append(f"System: {sys_text}")
    for role, text in _HISTORY.get(session_id, []):
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {text}")
    parts.append(f"User: {user_prompt.strip()}")
    parts.append("Assistant:")
    return "
".join(parts)
