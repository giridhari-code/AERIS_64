"""Dynamic max_tokens + temperature (Grok-style heuristics) — not hardcoded answers."""

from __future__ import annotations

import re

_LONG_CUES = re.compile(
    r"\b("
    r"summar(y|ize|ise)|summary|explain(\s+in\s+detail)?|detailed|"
    r"chapter|book|essay|article|report|compare|analysis|analyze|"
    r"step\s+by\s+step|tutorial|guide|write\s+(a\s+)?(long|full|complete)|"
    r"list\s+all|everything\s+about|comprehensive|overview|code|python|program"
    r")\b",
    re.I,
)

_SHORT_CUES = re.compile(
    r"^\s*("
    r"hi|hello|hey|yo|sup|thanks|thank\s*you|ok|okay|bye|good\s*morning|"
    r"good\s*night|namaste|who\s+are\s+you|what\s+is\s+your\s+name|"
    r"hello\s+sir|hi\s+sir"
    r")[\s\?\.!]*$",
    re.I,
)

_CREATIVE_CUES = re.compile(
    r"\b(story|poem|joke|creative|imagine|brainstorm|ideas?)\b",
    re.I,
)


def _prompt_kind(prompt: str) -> str:
    p = (prompt or "").strip()
    n_words = len(p.split())
    n_chars = len(p)
    if _SHORT_CUES.match(p) or n_words <= 3:
        return "short"
    if _CREATIVE_CUES.search(p):
        return "creative"
    if _LONG_CUES.search(p) or n_words >= 40 or n_chars >= 200:
        return "long"
    if n_words >= 15 or n_chars >= 80:
        return "medium"
    return "normal"


def adaptive_max_tokens(prompt: str, user_max: int | None = None, auto: bool = True) -> int:
    """Grok-like length: short asks → short budget; long/summary → higher (capped)."""
    if not auto:
        return max(1, min(int(user_max or 64), 8192))

    cap = 8192
    if user_max is not None and user_max > 0:
        cap = min(int(user_max), 8192)

    kind = _prompt_kind(prompt)
    budget = {
        "short": 48,
        "normal": 96,
        "medium": 256,
        "long": 1024,
        "creative": 512,
    }.get(kind, 96)

    if kind == "long":
        p = (prompt or "").lower()
        n_words = len((prompt or "").split())
        if n_words >= 80 or len(prompt or "") >= 500:
            budget = 2048
        if "book" in p or "chapter" in p or "summar" in p:
            budget = max(budget, 1536)

    return max(24, min(budget, cap))


def adaptive_temperature(
    prompt: str,
    user_temp: float | None = None,
    auto: bool = True,
) -> float:
    """
    Grok-like temperature:
      short/factual → low (focused)
      normal → mid-low
      long explain → slightly higher
      creative → higher
    user_temp acts as a soft preference blend when auto=True; hard override when auto=False.
    """
    if not auto:
        return max(0.0, min(float(user_temp if user_temp is not None else 0.2), 2.0))

    kind = _prompt_kind(prompt)
    base = {
        "short": 0.15,
        "normal": 0.25,
        "medium": 0.35,
        "long": 0.40,
        "creative": 0.70,
    }.get(kind, 0.25)

    if user_temp is not None:
        # light blend so UI slider still matters without freezing dynamics
        u = max(0.0, min(float(user_temp), 2.0))
        base = 0.65 * base + 0.35 * u

    return max(0.0, min(base, 2.0))


def adaptive_sampling(
    prompt: str,
    user_max: int | None = None,
    user_temp: float | None = None,
    auto: bool = True,
) -> tuple[int, float]:
    """Return (max_new_tokens, temperature) for one request."""
    return (
        adaptive_max_tokens(prompt, user_max=user_max, auto=auto),
        adaptive_temperature(prompt, user_temp=user_temp, auto=auto),
    )


def should_stop_generation(decoded_so_far: str, min_chars: int = 8) -> bool:
    t = decoded_so_far
    if len(t) < min_chars:
        return False
    if re.search(r"(?i)\n\s*user\s*:", t):
        return True
    if re.search(r"(?i)\n\s*assistant\s*:", t):
        return True
    if re.search(r"<\|end\|>|<\|user\|>", t):
        return True
    if "\n\n" in t and len(t) > 20:
        return True
    if len(t) >= 24 and re.search(r"[.!?]\s*$", t.rstrip()):
        return True
    if re.search(r"(.)\1{10,}", t):
        return True
    return False
