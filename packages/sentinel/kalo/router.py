"""KALO BOT intent router — route user text to the right child capability.

No canned demo answers. Classification is rule-based + cheap signals so the
mother agent can dispatch to Bhutu / Megha / Buddy / tools / dataset / chat
without hard-coding product copy as “intelligence”.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Route:
    target: str
    confidence: float
    reason: str


_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("buddy", re.compile(
        r"(?i)^(?:/--?\s*)?buddy\b|/buddy|--buddy|\bbuddy\s+(on|off|pet|egg)\b"
    ), "buddy command"),
    ("megha", re.compile(
        r"(?i)\b(megha|security\s+review|secure\s+code|incident\s+triage|"
        r"vuln(?:erability)?|hardening|owasp|auth(?:n|z)?\s+check)\b"
    ), "security / Megha"),
    ("dataset", re.compile(
        r"(?i)\b(dataset\s*builder|training\s+data|collect\s+(urls?|files?)|"
        r"token\s+budget|balance\s+dataset|export\s+(jsonl|dataset))\b"
    ), "dataset pipeline"),
    ("ajeebwork", re.compile(
        r"(?i)\b(ajeeb\s*work|multi[- ]?step\s+plan|file\s+jail|"
        r"batch[- ]process\s+files|sub[- ]?agent)\b"
    ), "AjeebWork"),
    ("kalofacts", re.compile(
        r"(?i)\b(kalofacts?|artifact|save\s+(this\s+)?(doc|file|note))\b"
    ), "artifacts"),
    ("tools", re.compile(
        r"(?i)\b(weather|search\s+the\s+web|web\s+search|calculate|what\s+time|"
        r"wikipedia|currency|fetch\s+url|run\s+(this\s+)?code)\b"
    ), "external tools"),
    ("bhutu", re.compile(
        r"(?i)\b(bhutu|metacognition|should\s+i\s+use\s+tools?|"
        r"tool\s+policy|allow\s+tools?)\b"
    ), "Bhutu policy"),
]


def route_intent(text: str) -> Route:
    t = (text or "").strip()
    if not t:
        return Route("chat", 0.2, "empty")
    for target, pat, reason in _RULES:
        if pat.search(t):
            return Route(target, 0.85, reason)
    # default: general chat through SENTINEL / AERIS_64 model path
    return Route("chat", 0.55, "general dialogue")
