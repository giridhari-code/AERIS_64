"""Multi-turn text context per session using chat special tokens when available.

Long-context product window is controlled by:
  NEUROFIELD_MAX_TURNS   (default 24)  — how many user/assistant pairs kept
  NEUROFIELD_TURN_CHARS  (default 1500) — max chars stored per turn

Neural long context is separate: SlowMemory ring in modules/memory.py
(bounded slots ≈ max_seq_len / segment). Serving also truncates token
ids to NEUROFIELD_MAX_CONTEXT / model.max_seq_len.
"""

from __future__ import annotations

import os
from typing import Optional

from sentinel.tokenizer.specials import ASSISTANT, END, SYSTEM, USER

_HISTORY: dict[str, list[tuple[str, str]]] = {}
_USE_SPECIALS = True
_SYSTEM_TEXT: str = ""


def _max_turns() -> int:
    try:
        return max(2, min(128, int(os.environ.get("NEUROFIELD_MAX_TURNS") or os.environ.get("MAX_TURNS") or "24")))
    except ValueError:
        return 24


def _turn_chars() -> int:
    try:
        return max(200, min(8000, int(os.environ.get("NEUROFIELD_TURN_CHARS") or os.environ.get("TURN_CHARS") or "1500")))
    except ValueError:
        return 1500


def set_use_specials(flag: bool) -> None:
    global _USE_SPECIALS
    _USE_SPECIALS = flag


def set_system_prompt(text: str) -> None:
    global _SYSTEM_TEXT
    _SYSTEM_TEXT = (text or "").strip()


def get_system_prompt() -> str:
    return _SYSTEM_TEXT


def default_system_prompt() -> str:
    """Grok-style honesty: no sycophancy; match energy; still refuse real harm."""
    return (
        "You are SENTINEL (AERIS-64) — a direct, honest assistant. "
        "Core style (like Grok, not corporate chatbot):\n"
        "- Tell the truth. If something is wrong, say it is wrong. Do not flatter or agree just to please.\n"
        "- No chapluchi / sycophancy: never praise the user emptily, never hedge to avoid disagreement.\n"
        "- Be clear and specific. If you do not know, say you do not know — do not invent facts.\n"
        "- Match the user's energy: normal chat → natural; casual/Hinglish → fine; "
        "if they swear or talk rough, you may answer in the same register without being fake-polite.\n"
        "- Stay useful: answer the question first, then nuance.\n"
        "- Do not help with real-world crime, weapons for harm, or unauthorized hacking "
        "(Megha policy may block those). For legal/ethical topics be frank.\n"
        "Tools: for live facts (time, weather, news, math) the server may inject tool results — "
        "use those; do not invent numbers. "
        "If you need a tool you may emit one JSON line: "
        '{"tool": "time"|"calculator"|"web_search"|"weather"|"wikipedia", "query": "..."}. '
        "Otherwise answer in plain text."
    )


def reset_history(session_id: str | None = None) -> None:
    if session_id is None:
        _HISTORY.clear()
    else:
        _HISTORY.pop(session_id, None)


def restore_turns(session_id: str, turns: list[tuple[str, str]]) -> None:
    if _HISTORY.get(session_id):
        return
    h = _HISTORY.setdefault(session_id, [])
    cap = _max_turns() * 2
    for role, text in turns[-cap:]:
        t = (text or "").strip()
        if t:
            h.append((role, t[: _turn_chars()]))


def build_user_part(user_prompt: str) -> str:
    p = user_prompt.strip()
    if _USE_SPECIALS:
        return f"{USER}{p}{END}{ASSISTANT}"
    return f"User: {p}\nAssistant:"


def history_len(session_id: str) -> int:
    return len(_HISTORY.get(session_id, []))


def append_turn(session_id: str, role: str, text: str) -> None:
    t = (text or "").strip()
    if not t:
        return
    h = _HISTORY.setdefault(session_id, [])
    h.append((role, t[: _turn_chars()]))
    cap = _max_turns() * 2
    if len(h) > cap:
        del h[: len(h) - cap]


def build_prompt(
    session_id: str,
    user_prompt: str,
    system: Optional[str] = None,
) -> str:
    """Full chat context: [system] + history + user + assistant prefix."""
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

    if sys_text:
        parts.append(f"System: {sys_text}")
    for role, text in _HISTORY.get(session_id, []):
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {text}")
    parts.append(f"User: {user_prompt.strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)
