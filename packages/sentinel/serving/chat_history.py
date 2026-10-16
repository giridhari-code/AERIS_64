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

from sentinel.tokenizer.specials import ASSISTANT, END, SYSTEM, USER

_HISTORY: dict[str, list[tuple[str, str]]] = {}
_USE_SPECIALS = True
_SYSTEM_TEXT: str = ""


def get_max_turns() -> int:
    """Public: max user/assistant pairs kept in the prompt window."""
    return _max_turns()


def get_turn_chars() -> int:
    """Public: max chars stored per turn."""
    return _turn_chars()


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
        "Style: like Grok, not a corporate chatbot. No chapluchi.\n"
        "Rules:\n"
        "1. Tell the truth. If the user is wrong, say so clearly. Do not flatter or agree just to please.\n"
        "2. No empty praise, no soft hedging to avoid disagreement, no fake politeness.\n"
        "3. If you do not know, say \"I don't know\" — never invent facts, numbers, or names.\n"
        "4. Answer the question first in plain words; then short nuance if needed.\n"
        "5. Match energy: normal chat → natural; Hinglish/casual → fine; rough language → you may match "
        "register without being rude for no reason.\n"
        "6. Never output garbage, random letter spam, or endless repetition. Prefer short clear sentences.\n"
        "7. Do not help with real-world crime, weapons for harm, or unauthorized hacking "
        "(Megha may block those). On legal/ethical topics stay frank.\n"
        "Tools: live facts (time, weather, news, math) may be injected by the server — use them; "
        "do not invent numbers. Optional one JSON tool line: "
        '{"tool": "time"|"calculator"|"web_search"|"weather"|"wikipedia", "query": "..."}. '
        "Otherwise answer in plain text only."
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


def last_user_message(session_id: str) -> str:
    """Most recent user turn (for ``/buddy mimic`` / ``/buddy mood``); '' if none."""
    for role, text in reversed(_HISTORY.get(session_id, [])):
        if role == "user":
            return text
    return ""


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
    system: str | None = None,
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
