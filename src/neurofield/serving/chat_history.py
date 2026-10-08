"""Multi-turn text context per session using chat special tokens when available."""

from __future__ import annotations

from neurofield.tokenizer.specials import ASSISTANT, END, USER

_HISTORY: dict[str, list[tuple[str, str]]] = {}
_MAX_TURNS = 6
_USE_SPECIALS = True  # set False to force legacy "User:" / "Assistant:" text


def set_use_specials(flag: bool) -> None:
    global _USE_SPECIALS
    _USE_SPECIALS = flag


def reset_history(session_id: str | None = None) -> None:
    if session_id is None:
        _HISTORY.clear()
    else:
        _HISTORY.pop(session_id, None)


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


def build_prompt(session_id: str, user_prompt: str) -> str:
    parts: list[str] = []
    for role, text in _HISTORY.get(session_id, []):
        if _USE_SPECIALS:
            tag = USER if role == "user" else ASSISTANT
            parts.append(f"{tag}{text}{END}")
        else:
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {text}")
    if _USE_SPECIALS:
        parts.append(f"{USER}{user_prompt.strip()}{END}")
        parts.append(ASSISTANT)
    else:
        parts.append(f"User: {user_prompt.strip()}")
        parts.append("Assistant:")
    return "".join(parts) if _USE_SPECIALS else "\n".join(parts)
