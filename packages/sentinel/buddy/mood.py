"""Infer Buddy animation mood from chat text (user or assistant).

Lightweight rules — no LLM. Supports emotional mimicry / "joker" energy matching.
"""

from __future__ import annotations

import re


def mood_from_text(text: str, *, role: str = "user", mimic: bool = True) -> str:
    """
    Map message content → animation id for the floating buddy.
    ``mimic=True`` (default): mirror user energy (joker-style).
    """
    t = (text or "").strip().lower()
    if not t:
        return "idle"

    if role == "assistant":
        if any(x in t for x in ("sorry", "error", "cannot", "can't", "unable", "blocked")):
            return "think"
        if any(x in t for x in ("!", "great", "done", "success", "😊", "🎉", "haan", "theek")):
            return "happy"
        return "talk"

    if re.search(r"\b(gussa|angry|mad|hate)\b", t) or re.search(r"[!]{2,}", t):
        return "excited" if mimic else "think"
    if re.search(r"\b(h+a+h+a|lol|lmao|😂|🤣|haha|hehe|mazak|joke|joker)\b", t):
        return "happy"
    if re.search(r"\b(udaas|sad|cry|rona|miss|lonely|tired|thak)\b", t):
        return "sleep" if mimic else "think"
    if re.search(r"\b(why|how|what|explain|soch|kyun|kaise|think|samajh)\b", t) or "?" in t:
        return "think"
    if re.search(r"\b(bye|good night|sleep|so ja|raat)\b", t):
        return "sleep"
    if re.search(r"\b(hi|hello|hey|namaste|wave|hola|salaam)\b", t):
        return "wave"
    if re.search(r"\b(wow|amazing|excited|yay|zindabad|op)\b", t):
        return "excited"
    if re.search(r"\b(pet|love|cute|pyaar)\b", t):
        return "pet"
    if re.search(r"\b(aeris|sentinel|buddy)\b", t):
        return "wave"
    if len(t) > 120:
        return "think"
    return "blink" if mimic else "idle"
