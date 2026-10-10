"""Route user prompts to the right tools and inject facts into the prompt."""

from __future__ import annotations

import os
import re
from typing import Optional

from neurofield.tools.calculator import calculate
from neurofield.tools.currency import convert_currency
from neurofield.tools.timeutil import now_report
from neurofield.tools.weather import weather_report
from neurofield.tools.web_search import web_search_report
from neurofield.tools.wikipedia import wikipedia_summary

_WEATHER = re.compile(
    r"\b(weather|temperature|forecast|barish|mausam|celsius|"
    r"kitna\s+(garam|thanda)|aaj\s+(ka\s+)?(mausam|weather)|today'?s?\s+weather)\b",
    re.I,
)
_SEARCH = re.compile(
    r"\b(search|google|look\s*up|latest\s+news|current\s+news|headline)\b",
    re.I,
)
_TIME = re.compile(
    r"\b(what\s+time|current\s+time|aaj\s+(ki\s+)?tarikh|today'?s?\s+date|"
    r"kitne\s+baje|date\s+today|time\s+now)\b",
    re.I,
)
_CALC = re.compile(
    r"(?:calculate|compute|what\s+is|kitna\s+hota)\s*[:\s]*([\d\.\s\+\-\*\/\^\(\)%]+)"
    r"|^\s*([\d\.\s\+\-\*\/\^\(\)%]{3,80})\s*\??\s*$",
    re.I,
)
_WIKI = re.compile(
    r"\b(wikipedia|who\s+is|what\s+is|tell\s+me\s+about|ke\s+bare\s+mein)\b",
    re.I,
)
_FX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(USD|INR|EUR|GBP|JPY|AED)\s*(?:to|in|→|->)\s*(USD|INR|EUR|GBP|JPY|AED)\b",
    re.I,
)
_PLACE = re.compile(
    r"\b(?:in|at|for)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
    re.I,
)
_CITIES = (
    "New Delhi", "Delhi", "Mumbai", "Bombay", "Kolkata", "Calcutta", "Bangalore",
    "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "London", "New York", "Tokyo", "Dubai", "Singapore", "Paris",
)


def tools_enabled() -> bool:
    return os.environ.get("NEUROFIELD_WEB_TOOLS", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _guess_place(prompt: str) -> str:
    p = prompt or ""
    for city in _CITIES:
        if re.search(rf"\b{re.escape(city)}\b", p, re.I):
            return city
    m = _PLACE.search(p)
    if m:
        cand = m.group(1).strip()
        if cand and not re.search(r"\b(kitna|hai|kya|what|today|aaj)\b", cand, re.I):
            return cand
    return "Delhi"


def _wiki_topic(prompt: str) -> str:
    p = re.sub(r"\?+$", "", prompt or "").strip()
    p = re.sub(
        r"^(who\s+is|what\s+is|tell\s+me\s+about|wikipedia)\s+",
        "",
        p,
        flags=re.I,
    )
    # fix typo in pattern
    return p.strip()[:80] or prompt[:80]


def enrich_prompt_with_tools(prompt: str) -> tuple[str, Optional[str]]:
    if not tools_enabled() or not (prompt or "").strip():
        return prompt, None

    p = prompt.strip()
    notes: list[str] = []

    if _TIME.search(p):
        tz = "Asia/Kolkata" if re.search(r"\b(india|ist|kolkata|delhi|mumbai)\b", p, re.I) else None
        notes.append(now_report(tz))

    if _WEATHER.search(p):
        notes.append(weather_report(_guess_place(p)))

    mfx = _FX.search(p)
    if mfx:
        notes.append(convert_currency(float(mfx.group(1)), mfx.group(2), mfx.group(3)))

    mc = _CALC.search(p)
    if mc:
        expr = (mc.group(1) or mc.group(2) or "").strip()
        if expr:
            notes.append(calculate(expr))

    if _WIKI.search(p) and not _WEATHER.search(p):
        notes.append(wikipedia_summary(_wiki_topic(p)))

    if _SEARCH.search(p) and not notes:
        notes.append(web_search_report(p, max_results=4))
    elif re.search(r"\b(aaj|today|abhi)\b", p, re.I) and re.search(
        r"\b(news|headline|price|stock|match)\b", p, re.I
    ):
        notes.append(web_search_report(p, max_results=4))

    if not notes:
        return prompt, None

    block = "\n".join(notes)
    enriched = (
        f"[Tool results — use these facts; do not invent numbers]\n{block}\n\n"
        f"User question: {p}"
    )
    return enriched, block
