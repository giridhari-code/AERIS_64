"""0z planner — deterministic multi-step plans (no code tools)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    id: int
    tool: str | None
    args: dict[str, Any]
    intent: str


@dataclass
class Plan:
    objective: str
    steps: list[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "steps": [
                {"id": s.id, "tool": s.tool, "args": s.args, "intent": s.intent}
                for s in self.steps
            ],
        }


_WEATHER = re.compile(r"\b(weather|mausam|temperature|forecast)\b", re.I)
_SEARCH = re.compile(r"\b(search|news|latest|look\s*up|google|research)\b", re.I)
_TIME = re.compile(r"\b(time|date|tarikh|kitne\s+baje|timezone)\b", re.I)
_CALC = re.compile(r"([\d\.\s\+\-\*\/\^\(\)%]{3,80})")
_WIKI = re.compile(r"\b(who\s+is|what\s+is|wikipedia|tell\s+me\s+about)\b", re.I)
_FX = re.compile(
    r"(\d+(?:\.\d+)?)\s*(USD|INR|EUR|GBP)\s*(?:to|in)\s*(USD|INR|EUR|GBP)",
    re.I,
)
_GIT = re.compile(r"\b(git\s+status|repo\s+status|git\s+log|git\s+diff)\b", re.I)
_FILE = re.compile(r"\b(find\s+file|search\s+file|where\s+is\s+.+file)\b", re.I)
_FETCH = re.compile(r"https?://[^\s]+", re.I)
_PLACE = re.compile(r"\b(?:in|at|for)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)")


def build_plan(user_message: str) -> Plan:
    """Build a short tool plan from the user message (regex router, company-stable)."""
    p = Plan(objective=user_message.strip())
    text = user_message.strip()
    steps: list[PlanStep] = []
    n = 1

    if re.search(r"\b(project|repo|codebase|workspace)\b.*(overview|structure|context)|what\s+is\s+this\s+(repo|project)", text, re.I):
        steps.append(PlanStep(n, "project_context", {}, "project awareness"))
        n += 1

    if re.search(r"\b(list\s+(dir|directory|files)|ls\b|show\s+files)\b", text, re.I):
        steps.append(PlanStep(n, "list_dir", {"path": "."}, "list workspace"))
        n += 1

    mread = re.search(r"\b(read|open|show)\s+(file\s+)?([\w./-]+\.\w+)", text, re.I)
    if mread:
        steps.append(PlanStep(n, "read_file", {"path": mread.group(3)}, "read file"))
        n += 1

    if _GIT.search(text):
        cmd = "status"
        if re.search(r"\blog\b", text, re.I):
            cmd = "log"
        elif re.search(r"\bdiff\b", text, re.I):
            cmd = "diff"
        steps.append(PlanStep(n, "git", {"command": cmd}, "read-only git"))
        n += 1

    if _FILE.search(text):
        q = re.sub(r"\b(find\s+file|search\s+file|where\s+is|file)\b", "", text, flags=re.I).strip(" :?-")
        steps.append(PlanStep(n, "file_search", {"query": q or text}, "locate files"))
        n += 1

    murl = _FETCH.search(text)
    if murl:
        steps.append(PlanStep(n, "web_fetch", {"url": murl.group(0)}, "fetch URL"))
        n += 1

    if _WEATHER.search(text):
        loc = "Delhi"
        m = _PLACE.search(text)
        if m:
            loc = m.group(1)
        steps.append(PlanStep(n, "weather", {"location": loc}, "live weather"))
        n += 1

    if _TIME.search(text):
        steps.append(PlanStep(n, "time", {"timezone": "Asia/Kolkata"}, "current time"))
        n += 1

    mfx = _FX.search(text)
    if mfx:
        steps.append(
            PlanStep(
                n,
                "currency",
                {"amount": float(mfx.group(1)), "from": mfx.group(2).upper(), "to": mfx.group(3).upper()},
                "FX convert",
            )
        )
        n += 1

    if _WIKI.search(text) and not _WEATHER.search(text):
        topic = re.sub(r"\b(who\s+is|what\s+is|wikipedia|tell\s+me\s+about)\b", "", text, flags=re.I).strip(" ?")
        steps.append(PlanStep(n, "wikipedia", {"topic": topic or text}, "encyclopedia"))
        n += 1

    if _SEARCH.search(text) and not steps:
        steps.append(PlanStep(n, "web_search", {"query": text, "max_results": 5}, "web research"))
        n += 1

    # bare arithmetic
    if re.search(r"^(calculate|compute|what\s+is)\b", text, re.I) or re.match(r"^[\d\.\s\+\-\*\/\(\)%]+$", text):
        expr = text
        m = re.search(r"(calculate|compute|what\s+is)\s*[:\s]*(.+)", text, re.I)
        if m:
            expr = m.group(2)
        steps.append(PlanStep(n, "calculator", {"expression": expr.strip()}, "calculate"))
        n += 1

    # default research if nothing matched but question-like
    if not steps and ("?" in text or len(text.split()) >= 4):
        steps.append(PlanStep(n, "web_search", {"query": text, "max_results": 4}, "general research"))

    p.steps = steps
    return p
