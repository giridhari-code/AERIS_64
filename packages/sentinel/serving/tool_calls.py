"""Parse model-emitted tool calls (JSON) and execute via unified dispatch."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from sentinel.tools.dispatch import run_tool


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
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
        query = obj.get("query") or obj.get("expr") or obj.get("q")
        if query is None:
            query = args.get("query") or args.get("expr") or args.get("q") or ""
        # merge flat fields into args
        merged = dict(args)
        for k, v in obj.items():
            if k in ("tool", "name", "arguments", "raw"):
                continue
            merged.setdefault(k, v)
        if query and "query" not in merged:
            merged["query"] = query
        return {
            "tool": str(name).lower().strip(),
            "query": str(query or "").strip(),
            "arguments": merged,
            "raw": raw,
        }
    return None


def run_tool_call(call: dict[str, Any]) -> str:
    """Execute a parsed tool call; always returns a string report."""
    name = (call.get("tool") or "").lower()
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if not args and call.get("query"):
        args = {"query": call["query"]}
    out = run_tool(name, args)
    if out.get("ok"):
        return str(out.get("result") or "")
    return f"Tool error ({name}): {out.get('error') or 'failed'}"
