"""Infer Buddy animation mood from chat text (user or assistant)."""

from __future__ import annotations

import re


def mood_from_text(text: str, *, role: str = "user") -> str:
    """
    Map message content → animation id for the floating buddy.
    Does not call the LLM; lightweight rules so UI stays instant.
    """
    t = (text or "").strip().lower()
    if not t:
        return "idle"

    if role == "assistant":
        # model is "talking"
        if any(x in t for x in ("sorry", "error", "cannot", "can't", "unable")):
            return "think"
        if any(x in t for x in ("!", "great", "done", "success", "😊", "🎉")):
            return "happy"
        return "talk"

    # user
    if re.search(r"\b(h+a+h+a|lol|lmao|😂|haha|hehe)\b", t) or "!" in t:
        return "happy"
    if re.search(r"\b(why|how|what|explain|soch|kyun|kaise|think)\b", t) or "?" in t:
        return "think"
    if re.search(r"\b(bye|good night|sleep|so ja)\b", t):
        return "sleep"
    if re.search(r"\b(hi|hello|hey|namaste|wave|hola)\b", t):
        return "wave"
    if re.search(r"\b(wow|amazing|excited|yay)\b", t):
        return "excited"
    if re.search(r"\b(pet|love|cute)\b", t):
        return "pet"
    if len(t) > 120:
        return "think"
    return "blink"
