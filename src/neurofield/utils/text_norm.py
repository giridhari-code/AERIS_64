"""Lightweight typo normalization for inference."""

from __future__ import annotations

import re

# Basic typo fixes only (English)
_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwhoami\b", re.I), "who am i"),
    (re.compile(r"\bhell+o\b", re.I), "hello"),
    (re.compile(r"\bhelo\b", re.I), "hello"),
    (re.compile(r"\bth?n?a?k\s*you\b", re.I), "thank you"),
    (re.compile(r"\bplz\b", re.I), "please"),
    (re.compile(r"\bpls\b", re.I), "please"),
]

def normalize_prompt(text: str) -> str:
    """Fix common typos before tokenization."""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    for pat, rep in _REPLACEMENTS:
        t = pat.sub(rep, t)
    return t
