"""Decide when to call web/weather tools and inject facts into the prompt."""

from __future__ import annotations

import os
import re
from typing import Optional

from neurofield.tools.weather import weather_report
from neurofield.tools.web_search import web_search_report

_WEATHER = re.compile(
    r"\b(weather|temperature|forecast|barish|mausam|°c|celsius|kitna\s+(garam|thanda)|"
    r"aaj\s+(ka\s+)?(mausam|weather)|today'?s?\s+weather)\b",
    re.I,
)
_SEARCH = re.compile(
    r"\b(search|google|look\s*up|latest|news|who\s+is|what\s+is|"
    r"current|price\s+of|score|election|wikipedia)\b",
    re.I,
)
_PLACE = re.compile(
    r"\b(?:in|at|for|of)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"
    r"|(?:weather|mausam)\s+(?:in\s+)?([A-Za-z][A-Za-z\s]{1,40}?)(?:\?|$)",
    re.I,
)


def tools_enabled() -> bool:
    return os.environ.get("NEUROFIELD_WEB_TOOLS", "1").strip() not in {"0", "false", "no"}


_CITIES = (
    "New Delhi", "Delhi", "Mumbai", "Bombay", "Kolkata", "Calcutta", "Bangalore",
    "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "London", "New York", "Tokyo", "Dubai", "Singapore", "Paris",
)


def _guess_place(prompt: str) -> str:
    p = prompt or ""
    for city in _CITIES:
        if re.search(rf"\b{re.escape(city)}\b", p, re.I):
            return city
    m = _PLACE.search(p)
    if m:
        cand = (m.group(1) or m.group(2) or "").strip()
        # reject garbage fragments
        if cand and len(cand) < 40 and not re.search(r"\b(kitna|hai|kya|what|today|aaj)\b", cand, re.I):
            return cand
    return "Delhi"


def enrich_prompt_with_tools(prompt: str) -> tuple[str, Optional[str]]:
    """
    Returns (possibly enriched prompt, tool_note).
    Tool facts are prepended so the small LM can condition on real data.
    """
    if not tools_enabled() or not (prompt or "").strip():
        return prompt, None

    p = prompt.strip()
    notes: list[str] = []

    if _WEATHER.search(p):
        place = _guess_place(p)
        notes.append(weather_report(place))

    # Always allow explicit search; also weather+news style
    if _SEARCH.search(p) and not notes:
        notes.append(web_search_report(p, max_results=4))
    elif _SEARCH.search(p) and notes:
        # weather already ran; optional extra search skipped to save latency
        pass
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
