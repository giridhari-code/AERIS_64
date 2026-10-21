"""Route user prompts to the right tools and inject facts into the prompt."""

from __future__ import annotations

import os
import re
from typing import Optional

from sentinel.tools.calculator import calculate
from sentinel.tools.currency import convert_currency
from sentinel.tools.git_tool import git_report
from sentinel.tools.timeutil import now_report
from sentinel.tools.weather import weather_report
from sentinel.tools.web_search import web_search_report
from sentinel.tools.wikipedia import wikipedia_summary
from sentinel.tools.rag_tool import rag_retrieve
from sentinel.tools.tool_search import tool_search_report, search_tools, format_tool_catalog_for_prompt
from sentinel.tools.code_execution import code_execution
from sentinel.tools.bash_tool import bash
from sentinel.tools.web_fetch import web_fetch
from sentinel.tools.file_search import file_search
from sentinel.tools.megha_tool import megha_report, megha_capabilities_text

_WEATHER = re.compile(
    r"\b(weather|temperature|forecast|barish|mausam|celsius|"
    r"kitna\s+(garam|thanda)|aaj\s+(ka\s+)?(mausam|weather)|today'?s?\s+weather)\b",
    re.I,
)
_SEARCH = re.compile(
    r"\b(search|google|look\s*up|latest\s+news|current\s+news|headline|"
    r"news|khabar|kya\s+chal\s+raha|trending|twitter|\bx\b\s+(pe|par|on)|"
    r"what.?s\s+happening|break(?:ing)?\s+news)\b",
    re.I,
)
_TIME = re.compile(
    r"\b(what\s+time|current\s+time|aaj\s+(ki\s+)?tarikh|today'?s?\s+date|"
    r"kitne\s+baje|date\s+today|time\s+now|abhi\s+kitne|samay\s+kya|"
    r"clock|timezone)\b",
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
_RAG = re.compile(
    r"\b(rag|retrieve|knowledge\s*base|from\s+(the\s+)?docs?|according\s+to\s+(our|the)\s+(docs?|kb))\b",
    re.I,
)

_CODE_EXEC = re.compile(
    r"\b(run\s+(this\s+)?(code|python)|execute\s+(this\s+)?(code|python)|"
    r"code\s+interpreter|python\s+code)\b",
    re.I,
)
_BASH = re.compile(
    r"\b(bash|shell\s+command|run\s+(this\s+)?(command|shell)|terminal)\b",
    re.I,
)
_WEB_FETCH = re.compile(
    r"\b(fetch\s+(this\s+)?(url|page|link)|open\s+(this\s+)?(url|link)|"
    r"read\s+(this\s+)?(page|url|website))\b",
    re.I,
)
_FILE_SEARCH = re.compile(
    r"\b(find\s+file|search\s+file|file\s+search|where\s+is\s+the\s+file|"
    r"locate\s+file)\b",
    re.I,
)

_TOOL_SEARCH = re.compile(
    r"\b(tool\s*search|search\s*tools|which\s+tools?|available\s+tools?|"
    r"kya\s+tools?\s+(hain|hai)|tools?\s+list)\b",
    re.I,
)
_GIT = re.compile(
    r"\b(git\s+(status|log|diff|add|commit|branch|show)|"
    r"repo\s+status|commit\s+(kar|the|this)|push\s+to\s+git|"
    r"what('?s|\s+is)\s+the\s+(git\s+)?status)\b",
    re.I,
)
_MEGHA = re.compile(
    r"\b(megha|security\s+(checklist|scan|review|harden|hardening)|"
    r"owasp|secure\s+code\s+review|incident\s+triage|"
    r"authorized\s+(pentest|assessment)|defensive\s+monitor)\b",
    re.I,
)
_PLACE = re.compile(  # preposition is case-insensitive; the place name must be Capitalised
    r"\b(?i:in|at|for)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
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


def enrich_prompt_with_tools(
    prompt: str,
    allowed_tools: Optional[set[str] | tuple[str, ...] | list[str]] = None,
) -> tuple[str, Optional[str]]:
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

    if _RAG.search(p):
        notes.append(rag_retrieve(p, top_k=4))

    if _MEGHA.search(p):
        if re.search(r"\b(capabilities|modes|help)\b", p, re.I):
            notes.append(megha_capabilities_text())
        else:
            notes.append(megha_report(p, mode="auto"))

    if _TOOL_SEARCH.search(p):
        notes.append(tool_search_report(p, top_k=6))

    if _GIT.search(p):
        # Map natural language → safe git subcommand
        if re.search(r"\b(log|history|commits)\b", p, re.I):
            notes.append(git_report("log"))
        elif re.search(r"\b(diff|changes|changed)\b", p, re.I):
            notes.append(git_report("diff"))
        elif re.search(r"\b(branch|branches)\b", p, re.I):
            notes.append(git_report("branch"))
        else:
            notes.append(git_report("status"))

    if _SEARCH.search(p) and not notes:
        notes.append(web_search_report(p, max_results=4))
    elif re.search(r"\b(aaj|today|abhi)\b", p, re.I) and re.search(
        r"\b(news|headline|price|stock|match)\b", p, re.I
    ):
        notes.append(web_search_report(p, max_results=4))


    def _allow(name: str) -> bool:
        if allowed_tools is None:
            return True
        return name in set(allowed_tools)

    if _CODE_EXEC.search(p) and _allow("code_execution"):
        mcode = re.search(r"```(?:python)?\s*([\s\S]+?)```", p)
        code = mcode.group(1).strip() if mcode else ""
        if code:
            notes.append(code_execution(code))
        else:
            notes.append("CodeExecution: provide Python code in a ```python block.")

    if _BASH.search(p) and not _CODE_EXEC.search(p) and _allow("bash"):
        mcmd = re.search(r"`([^`]+)`", p)
        cmd = mcmd.group(1).strip() if mcmd else ""
        if cmd:
            notes.append(bash(cmd))
        else:
            notes.append("Bash: provide a command in backticks, e.g. `ls -la`.")

    if _WEB_FETCH.search(p):
        murl = re.search(r"https?://[^\s\]\)]+", p)
        if murl:
            notes.append(web_fetch(murl.group(0)))
        else:
            notes.append("WebFetch: provide a full http(s) URL.")

    if _FILE_SEARCH.search(p):
        # Use remaining text after trigger words as query
        q = re.sub(
            r"\b(find\s+file|search\s+file|file\s+search|where\s+is\s+the\s+file|locate\s+file)\b",
            "",
            p,
            flags=re.I,
        ).strip(" :?-")
        if q:
            notes.append(file_search(q))

    # Open "give me news" without other tools
    if not notes and re.search(
        r"\b(news|khabar|headlines?|kya\s+chal\s+raha|update\s+do|batao\s+news)\b",
        p,
        re.I,
    ):
        notes.append(web_search_report(p, max_results=5))

    if not notes:
        return prompt, None

    block = "\n".join(notes)
    enriched = (
        f"[Tool results — use these facts; do not invent numbers]\n{block}\n\n"
        f"User question: {p}"
    )
    return enriched, block
