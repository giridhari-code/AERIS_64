"""Infer Buddy animation mood from chat text (user or assistant).

Lightweight rules — no LLM. Two layers:
  * ``mood_from_text`` — message → float animation id (emotional mimicry)
  * ``sense_mood``     — message → emotion label for ``/buddy mood``
"""

from __future__ import annotations

import re

# emotion → (emoji, buddy line, float animation) — used by /buddy mood
MOOD_META: dict[str, tuple[str, str, str]] = {
    "happy": ("😊", "khushi clear dikh rahi hai — buddy bhi khil gaya!", "happy"),
    "excited": ("🤩", "full energy — buddy jump kar raha hai!", "excited"),
    "angry": ("😤", "thoda gussa hai… main shaant rehta hoon, aap bolo.", "think"),
    "sad": ("🫂", "udaas lag raha hai — main yahin hoon, saath hoon.", "sleep"),
    "tired": ("😴", "thakaan dikh rahi hai — rest bhi zaroori hai.", "sleep"),
    "loving": ("💛", "pyaar wali baat — buddy ka dil bhi garam ho gaya.", "pet"),
    "curious": ("🤔", "kuch samajhna hai — achha sawaal.", "think"),
    "neutral": ("🙂", "bas normal — main dhyan se sun raha hoon.", "blink"),
}

_EMOTIONS: list[tuple[str, str]] = [
    ("angry", r"\b(gussa|angry|mad|hate|krodh|bakwas|irritat)\b"),
    ("sad", r"\b(udaas|sad|cry|rona|lonely|dukhi|miss|akela|bura\s+lag)\b"),
    ("tired", r"\b(tired|thak|neend|sleepy|bore)\b"),
    ("excited", r"\b(wow|amazing|excited|yay|zindabad|op|chalo|lets\s+go|let's\s+go)\b"),
    ("loving", r"\b(pet|love|cute|pyaar|pyar|jaan|dil)\b"),
    ("happy", r"\b(khush|happy|maza|badhiya|zabardast|shukriya|thanks|dhanyavad|great)\b|[!]{2,}|[😊🎉😂🤣]"),
    ("curious", r"\b(why|how|what|explain|kyun|kaise|samajh|soch|think)\b|\?"),
]


def sense_mood(text: str) -> str:
    """Coarse emotion label for a message (``happy``/``sad``/…/``neutral``)."""
    t = (text or "").strip().lower()
    if not t:
        return "neutral"
    for label, pat in _EMOTIONS:
        if re.search(pat, t):
            return label
    return "neutral"


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

    if re.search(r"\b(gussa|angry|mad|hate|krodh)\b", t) or re.search(r"[!]{2,}", t):
        return "excited" if mimic else "think"
    if re.search(r"\b(h+a+h+a|lol|lmao|😂|🤣|haha|hehe|mazak|joke|joker)\b", t):
        return "happy"
    if re.search(r"\b(khush|maza|badhiya|zabardast|shukriya|dhanyavad|thanks)\b", t):
        return "happy"
    if re.search(r"\b(tension|pareshan|stress|chinta|worried)\b", t):
        return "think"
    if re.search(r"\b(udaas|sad|cry|rona|miss|lonely|tired|thak|neend)\b", t):
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
