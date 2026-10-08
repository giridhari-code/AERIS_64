"""Light post-process only — NO canned/hardcoded answers.

Tiny models still need more training data for quality.
This module only cleans obvious generation artifacts.
"""

from __future__ import annotations

import re


def polish_reply(prompt: str, text: str) -> str:
    """Strip role-leak and extreme repetition. Do not replace with templates."""
    del prompt  # unused — model must answer, not a lookup table
    t = (text or "").strip()
    if not t:
        return t

    # Remove leaked train-format prefixes only
    t = re.sub(r"(?i)^\s*(user|assistant|asse)\s*:\s*", "", t)
    t = re.sub(r"(?i)\n\s*(user|assistant)\s*:\s*", "\n", t)

    # Collapse extreme character runs (llllllll → lll)
    t = re.sub(r"(.)\1{6,}", lambda m: m.group(1) * 3, t)

    return t.strip()
