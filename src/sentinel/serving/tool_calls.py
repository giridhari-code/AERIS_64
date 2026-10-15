"""Parse model-emitted tool calls (JSON) and execute via tools package."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from sentinel.tools.calculator import calculate
from sentinel.tools.timeutil import now_report
from sentinel.tools.weather import weather_report
from sentinel.tools.web_search import web_search_report
from sentinel.tools.wikipedia import wikipedia_summary


def extract_tool_call(text: str) -> Optional[dict[str, Any]]:
    """Return first tool call dict from model text, or None."""
    if not text or "{" not in text:
        return None
    for m in re.finditer(r"\{[^{}]+\}", text):
        raw = m.group(0)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name")
        if not name:
            continue
        query = obj.get("query") or obj.get("expr") or obj.get("q")
        if query is None and isinstance(obj.get("arguments"), dict):
            args = obj["arguments"]
            query = args.get("query") or args.get("expr") or args.get("q") or ""
        return {
            "tool": str(name).lower().strip(),
            "query": str(query or "").strip(),
            "raw": raw,
        }
    return None


def run_tool_call(call: dict[str, Any]) -> str:
    """Execute a parsed tool call; always returns a string report."""
    name = (call.get("tool") or "").lower()
    q = call.get("query") or ""
    try:
        if name in ("time", "datetime", "clock"):
            return now_report()
        if name in ("calculator", "calc", "math"):
            return calculate(q or "0")
        if name in ("web_search", "search", "news"):
            return web_search_report(q or "news", max_results=4)
        if name in ("weather",):
            return weather_report(q or "local")
        if name in ("wikipedia", "wiki"):
            return wikipedia_summary(q or "SENTINEL")
        return f"Unknown tool: {name}"
    except Exception as e:  # noqa: BLE001
        return f"Tool error ({name}): {type(e).__name__}: {e}"
