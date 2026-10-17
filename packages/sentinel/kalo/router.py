"""KALO BOT intent router — deterministic, no LLM, no demo replies."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    target: str
    confidence: float
    reason: str


# Order matters: first match wins.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "buddy",
        re.compile(
            r"(?i)(?:^|[\s])(?:/--?\s*)?buddy\b|/buddy|--buddy"
        ),
        "buddy command",
    ),
    (
        "megha",
        re.compile(
            r"(?i)\b(megha|security\s+review|secure\s+code|incident\s+triage|"
            r"vuln(?:erability)?|hardening|owasp|auth(?:n|z)?\s+check|"
            r"defensive\s+security)\b"
        ),
        "security / Megha",
    ),
    (
        "dataset",
        re.compile(
            r"(?i)\b(dataset(?:\s*builder)?|training\s+data|training\s+jsonl|"
            r"collect\s+(urls?|files?|paths?)|token\s+(budget|count|progress)|"
            r"balance\s+(the\s+)?dataset|export\s+(jsonl|csv|txt|dataset|train)|"
            r"\bimbalance\b|\b1m\s*token|\bcategory\s+budget)\b"
        ),
        "dataset pipeline",
    ),
    (
        "ajeebwork",
        re.compile(
            r"(?i)\b(ajeeb\s*work|multi[- ]?step\s+plan|file\s+jail|"
            r"batch[- ]process\s+files|sub[- ]?agent)\b"
        ),
        "AjeebWork",
    ),
    (
        "kalofacts",
        re.compile(
            r"(?i)\b(kalofacts?|artifact\s+workspace|save\s+(this\s+)?(as\s+)?(artifact|doc|note))\b"
        ),
        "artifacts",
    ),
    (
        "tools",
        re.compile(
            r"(?i)\b(weather|forecast|barish|mausam|search\s+the\s+web|web\s+search|"
            r"google|calculate|compute|what\s+time|current\s+time|wikipedia|"
            r"currency|convert\s+\d|fetch\s+url|run\s+(this\s+)?(code|python))\b"
        ),
        "external tools",
    ),
    (
        "bhutu",
        re.compile(
            r"(?i)\b(bhutu|metacognition|tool\s+policy|allow\s+tools?|"
            r"should\s+i\s+use\s+tools?)\b"
        ),
        "Bhutu policy",
    ),
]


def route_intent(text: str) -> Route:
    t = (text or "").strip()
    if not t:
        return Route("chat", 0.2, "empty")
    for target, pat, reason in _RULES:
        if pat.search(t):
            return Route(target, 0.9, reason)
    return Route("chat", 0.55, "general dialogue")
