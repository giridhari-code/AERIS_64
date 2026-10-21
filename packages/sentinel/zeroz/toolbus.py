"""0z tool bus — explicit, allow-listed, audited tool dispatch."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sentinel.zeroz.policy import ALLOWED_TOOLS, BLOCKED_TOOLS, check_tool_name

logger = logging.getLogger("zeroz.toolbus")

ToolFn = Callable[[dict[str, Any]], str]


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: ToolFn
    parameters: dict[str, Any]


@dataclass
class ToolCallResult:
    name: str
    ok: bool
    output: str
    ms: float
    error: Optional[str] = None


def _handlers() -> dict[str, ToolSpec]:
    """Lazy bind to existing SENTINEL tools (no coding tools)."""
    specs: dict[str, ToolSpec] = {}

    def add(name: str, desc: str, fn: ToolFn, params: dict[str, Any]) -> None:
        if name in BLOCKED_TOOLS:
            return
        specs[name] = ToolSpec(name, desc, fn, params)

    def weather(args: dict[str, Any]) -> str:
        from sentinel.tools.weather import weather_report
        return weather_report(str(args.get("location") or args.get("query") or "Delhi"))

    def web_search(args: dict[str, Any]) -> str:
        from sentinel.tools.web_search import web_search_report
        return web_search_report(str(args.get("query") or ""), max_results=int(args.get("max_results") or 5))

    def web_fetch(args: dict[str, Any]) -> str:
        from sentinel.tools.web_fetch import web_fetch
        return web_fetch(str(args.get("url") or ""))

    def calculator(args: dict[str, Any]) -> str:
        from sentinel.tools.calculator import calculate
        return calculate(str(args.get("expression") or args.get("query") or "0"))

    def time_now(args: dict[str, Any]) -> str:
        from sentinel.tools.timeutil import now_report
        return now_report(str(args.get("timezone") or "") or None)

    def wikipedia(args: dict[str, Any]) -> str:
        from sentinel.tools.wikipedia import wikipedia_summary
        return wikipedia_summary(str(args.get("topic") or args.get("query") or ""), lang=str(args.get("lang") or "en"))

    def currency(args: dict[str, Any]) -> str:
        from sentinel.tools.currency import convert_currency
        return convert_currency(
            float(args.get("amount") or 1),
            str(args.get("from") or args.get("from_currency") or "USD"),
            str(args.get("to") or args.get("to_currency") or "INR"),
        )

    def git(args: dict[str, Any]) -> str:
        from sentinel.tools.git_tool import git_report
        cmd = str(args.get("command") or "status").strip().lower()
        # 0z: read-only git only
        if cmd in ("push", "commit", "add", "rebase", "reset", "merge"):
            return "0z code: git write operations are disabled. Use status|log|diff|branch|show."
        return git_report(cmd)

    def file_search(args: dict[str, Any]) -> str:
        from sentinel.tools.file_search import file_search as fs
        return fs(str(args.get("query") or ""))

    def rag(args: dict[str, Any]) -> str:
        try:
            from sentinel.rag.meta_rag import auto_rag
            prompt, info = auto_rag(str(args.get("query") or ""), session_key="zeroz")
            return f"RAG info={info}\n\n{prompt[:3000]}"
        except Exception as e:
            return f"RAG unavailable: {type(e).__name__}: {e}"

    def tool_search(args: dict[str, Any]) -> str:
        from sentinel.tools.tool_search import tool_search_report
        return tool_search_report(str(args.get("query") or "available"))

    add("weather", "Current weather for a place", weather, {"location": "string"})
    add("web_search", "Search the public web", web_search, {"query": "string", "max_results": "int"})
    add("web_fetch", "Fetch a public URL", web_fetch, {"url": "string"})
    add("calculator", "Arithmetic expression", calculator, {"expression": "string"})
    add("time", "Current date/time", time_now, {"timezone": "string"})
    add("wikipedia", "Encyclopedia summary", wikipedia, {"topic": "string", "lang": "string"})
    add("currency", "FX conversion", currency, {"amount": "number", "from": "string", "to": "string"})
    add("git", "Read-only git status/log/diff/branch", git, {"command": "string"})
    add("file_search", "Find files by name/query", file_search, {"query": "string"})
    add("rag", "Retrieve from knowledge base", rag, {"query": "string"})
    add("tool_search", "List/search available tools", tool_search, {"query": "string"})

    def list_dir_tool(args: dict[str, Any]) -> str:
        from sentinel.zeroz.workspace import list_dir, default_root
        return list_dir(str(args.get("path") or "."), root=default_root())

    def read_file_tool(args: dict[str, Any]) -> str:
        from sentinel.zeroz.workspace import read_text_file
        return read_text_file(str(args.get("path") or ""))

    def project_context(args: dict[str, Any]) -> str:
        from sentinel.zeroz.context import load_project_context
        return load_project_context()

    add("list_dir", "List directory under workspace", list_dir_tool, {"path": "string"})
    add("read_file", "Read a text file (read-only)", read_file_tool, {"path": "string"})
    add("project_context", "Summarize project root + key config files", project_context, {})

    return specs


class ToolBus:
    def __init__(self, allowed: Optional[set[str]] = None):
        self._specs = _handlers()
        allow = set(allowed) if allowed is not None else set(ALLOWED_TOOLS)
        self.allowed = {n for n in allow if n in self._specs and n not in BLOCKED_TOOLS}

    def schemas(self) -> list[dict[str, Any]]:
        out = []
        for name in sorted(self.allowed):
            sp = self._specs[name]
            out.append({
                "name": sp.name,
                "description": sp.description,
                "parameters": sp.parameters,
            })
        return out

    def call(self, name: str, arguments: Optional[dict[str, Any]] = None) -> ToolCallResult:
        decision = check_tool_name(name)
        t0 = time.perf_counter()
        if not decision.allow or name not in self.allowed:
            return ToolCallResult(
                name=name,
                ok=False,
                output=decision.public_message or f"tool blocked: {name}",
                ms=(time.perf_counter() - t0) * 1000,
                error="policy",
            )
        spec = self._specs.get(name)
        if spec is None:
            return ToolCallResult(name, False, f"unknown tool: {name}", 0.0, "unknown")
        try:
            out = spec.handler(arguments or {})
            ms = (time.perf_counter() - t0) * 1000
            text = (out or "").strip()
            ok = bool(text) and not text.lower().startswith("error")
            return ToolCallResult(name, ok, text[:8000], ms, None if ok else "empty_or_error")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            logger.exception("tool %s failed", name)
            return ToolCallResult(name, False, f"{type(e).__name__}: {e}", ms, str(e))
