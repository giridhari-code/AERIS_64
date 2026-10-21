"""Unified tool dispatch for backend API and model tool-calls.

All catalog tools route through ``run_tool(name, arguments)``.
Sandbox-sensitive tools (bash/code) still respect existing env gates when called
via dedicated sandbox routes; this dispatcher returns clear errors if disabled.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from sentinel.tools.tiers import deny_message, is_allowed, tier_of

logger = logging.getLogger("sentinel.tools.dispatch")

Args = dict[str, Any]
Handler = Callable[[Args], str]


def _s(args: Args, *keys: str, default: str = "") -> str:
    for k in keys:
        v = args.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _f(args: Args, key: str, default: float = 0.0) -> float:
    try:
        return float(args.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(args: Args, key: str, default: int = 0) -> int:
    try:
        return int(args.get(key, default))
    except (TypeError, ValueError):
        return default


def _handlers() -> dict[str, Handler]:
    from sentinel.tools.calculator import calculate
    from sentinel.tools.currency import convert_currency
    from sentinel.tools.file_search import file_search
    from sentinel.tools.git_tool import git_report
    from sentinel.tools.timeutil import now_report
    from sentinel.tools.weather import weather_report
    from sentinel.tools.web_fetch import web_fetch
    from sentinel.tools.web_search import web_search_report
    from sentinel.tools.wikipedia import wikipedia_summary
    from sentinel.tools.tool_search import tool_search_report

    def weather(a: Args) -> str:
        return weather_report(_s(a, "location", "query", "place", default="Delhi"))

    def web_search(a: Args) -> str:
        return web_search_report(
            _s(a, "query", "q", default="news"),
            max_results=_i(a, "max_results", 5),
        )

    def time_now(a: Args) -> str:
        tz = _s(a, "timezone", "tz") or None
        return now_report(tz)

    def calculator(a: Args) -> str:
        return calculate(_s(a, "expression", "query", "expr", default="0"))

    def wikipedia(a: Args) -> str:
        return wikipedia_summary(
            _s(a, "topic", "query", default=""),
            lang=_s(a, "lang", default="en") or "en",
        )

    def currency_convert(a: Args) -> str:
        return convert_currency(
            _f(a, "amount", 1.0),
            _s(a, "from", "from_currency", default="USD") or "USD",
            _s(a, "to", "to_currency", default="INR") or "INR",
        )

    def git(a: Args) -> str:
        return git_report(_s(a, "command", "query", default="status") or "status")

    def rag_retrieve(a: Args) -> str:
        try:
            from sentinel.rag.meta_rag import auto_rag

            prompt, info = auto_rag(_s(a, "query", default=""), session_key="api")
            return f"rag info={info}\n\n{prompt[:4000]}"
        except Exception as e:  # noqa: BLE001
            return f"RAG error: {type(e).__name__}: {e}"

    def tool_search(a: Args) -> str:
        return tool_search_report(_s(a, "query", default="available"))

    def code_execution(a: Args) -> str:
        from sentinel.tools.code_execution import code_execution as ce

        return ce(_s(a, "code", "query", default=""))

    def bash(a: Args) -> str:
        from sentinel.tools.bash_tool import bash as b

        return b(_s(a, "command", "query", default=""))

    def web_fetch_h(a: Args) -> str:
        return web_fetch(_s(a, "url", "query", default=""))

    def file_search_h(a: Args) -> str:
        return file_search(_s(a, "query", default=""))

    def image_generation(a: Args) -> str:
        from sentinel.tools.image_generation import image_generation as ig

        return ig(_s(a, "prompt", "query", default=""))

    def computer_use(a: Args) -> str:
        from sentinel.tools.computer_use import computer_use as cu

        return cu(_s(a, "action", "query", default="help"))

    def megha(a: Args) -> str:
        from sentinel.tools.megha_tool import megha_report

        return megha_report(
            text=_s(a, "text", "query", default=""),
            mode=_s(a, "mode", default="auto") or "auto",
            target_type=_s(a, "target_type", default="web") or "web",
            authorized=bool(a.get("authorized", False)),
            scope=_s(a, "scope", default=""),
            attestor=_s(a, "attestor", default=""),
            framework=_s(a, "framework", default="general") or "general",
        )

    def dataset_builder(a: Args) -> str:
        from sentinel.tools.dataset_builder import dataset_builder_report

        paths = a.get("paths") if isinstance(a.get("paths"), list) else None
        urls = a.get("urls") if isinstance(a.get("urls"), list) else None
        return dataset_builder_report(
            action=_s(a, "action", default="progress") or "progress",
            paths=paths,
            urls=urls,
            category=_s(a, "category", default="general_language") or "general_language",
            license_note=_s(a, "license_note", default="unknown") or "unknown",
            export_format=_s(a, "export_format", default="jsonl") or "jsonl",
            work_dir=_s(a, "work_dir", default="") or None,
            text=_s(a, "text", default="") or None,
        )

    # AjeebWork
    def ajeebwork_file(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_file as fn

        return fn(
            action=_s(a, "action", default="read") or "read",
            path=_s(a, "path", default=""),
            content=_s(a, "content", default=""),
            old=_s(a, "old", default=""),
            new=_s(a, "new", default=""),
        )

    def ajeebwork_plan(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_plan as fn

        return fn(_s(a, "objective", "query", default=""), _s(a, "steps_json", default="[]") or "[]")

    def ajeebwork_subagents(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_subagents as fn

        return fn(_s(a, "tasks_json", default="[]") or "[]")

    def ajeebwork_task(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_task as fn

        return fn(
            action=_s(a, "action", default="list") or "list",
            name=_s(a, "name", default=""),
            command=_s(a, "command", default=""),
            task_id=_s(a, "task_id", default=""),
        )

    def ajeebwork_memory(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_memory as fn

        return fn(
            action=_s(a, "action", default="get") or "get",
            session_id=_s(a, "session_id", default="default") or "default",
            key=_s(a, "key", default=""),
            value=_s(a, "value", default=""),
        )

    def ajeebwork_document(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_document as fn

        return fn(
            kind=_s(a, "kind", default="markdown") or "markdown",
            path=_s(a, "path", default=""),
            title=_s(a, "title", default=""),
            body=_s(a, "body", default=""),
        )

    def ajeebwork_draft_edit(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_draft_edit as fn

        return fn(
            _s(a, "path", default=""),
            _s(a, "selected_text", default=""),
            _s(a, "replacement", default=""),
        )

    def ajeebwork_skill(a: Args) -> str:
        from sentinel.tools.ajeebwork_tools import ajeebwork_skill as fn

        return fn(
            action=_s(a, "action", default="list") or "list",
            name=_s(a, "name", default=""),
            content=_s(a, "content", default=""),
        )

    # KaloFacts
    def kalofact_create(a: Args) -> str:
        from sentinel.tools.kalofacts_tools import kalofact_create as fn

        return fn(
            title=_s(a, "title", default=""),
            content=_s(a, "content", "query", default=""),
            session_id=_s(a, "session_id", default=""),
        )

    def kalofact_get(a: Args) -> str:
        from sentinel.tools.kalofacts_tools import kalofact_get as fn

        return fn(_s(a, "artifact_id", "id", default=""), _s(a, "view", default=""))

    def kalofact_list(a: Args) -> str:
        from sentinel.tools.kalofacts_tools import kalofact_list as fn

        return fn(_s(a, "session_id", default=""), _i(a, "limit", 20))

    def kalofact_edit(a: Args) -> str:
        from sentinel.tools.kalofacts_tools import kalofact_edit as fn

        return fn(
            artifact_id=_s(a, "artifact_id", "id", default=""),
            content=_s(a, "content", default=""),
            title=_s(a, "title", default=""),
        )

    def kalofact_export(a: Args) -> str:
        from sentinel.tools.kalofacts_tools import kalofact_export as fn

        return fn(_s(a, "artifact_id", "id", default=""), _s(a, "dest_path", default=""))

    def kalofact_share(a: Args) -> str:
        from sentinel.tools.kalofacts_tools import kalofact_share as fn

        return fn(
            _s(a, "artifact_id", default=""),
            _s(a, "token", default=""),
            _s(a, "action", default="create") or "create",
        )

    m: dict[str, Handler] = {
        "web_search": web_search,
        "weather": weather,
        "time_now": time_now,
        "time": time_now,
        "calculator": calculator,
        "calc": calculator,
        "wikipedia": wikipedia,
        "wiki": wikipedia,
        "currency_convert": currency_convert,
        "currency": currency_convert,
        "git": git,
        "rag_retrieve": rag_retrieve,
        "rag": rag_retrieve,
        "tool_search": tool_search,
        "code_execution": code_execution,
        "code_interpreter": code_execution,
        "bash": bash,
        "web_fetch": web_fetch_h,
        "file_search": file_search_h,
        "image_generation": image_generation,
        "computer_use": computer_use,
        "megha": megha,
        "dataset_builder": dataset_builder,
        "ajeebwork_file": ajeebwork_file,
        "ajeebwork_plan": ajeebwork_plan,
        "ajeebwork_subagents": ajeebwork_subagents,
        "ajeebwork_task": ajeebwork_task,
        "ajeebwork_memory": ajeebwork_memory,
        "ajeebwork_document": ajeebwork_document,
        "ajeebwork_draft_edit": ajeebwork_draft_edit,
        "ajeebwork_skill": ajeebwork_skill,
        "kalofact_create": kalofact_create,
        "kalofact_get": kalofact_get,
        "kalofact_list": kalofact_list,
        "kalofact_edit": kalofact_edit,
        "kalofact_export": kalofact_export,
        "kalofact_share": kalofact_share,
    }
    return m


def known_tool_names() -> list[str]:
    return sorted(set(_handlers().keys()))


def run_tool(name: str, arguments: Optional[Args] = None) -> dict[str, Any]:
    """Execute one tool. Returns {ok, name, tier, latency_ms, result|error}."""
    n = (name or "").strip().lower()
    args = arguments or {}
    t0 = time.perf_counter()
    handlers = _handlers()
    fn = handlers.get(n)
    if fn is None:
        return {
            "ok": False,
            "name": n,
            "tier": tier_of(n),
            "latency_ms": 0.0,
            "error": f"unknown tool: {n}",
            "known": known_tool_names(),
        }
    if not is_allowed(n):
        return {
            "ok": False,
            "name": n,
            "tier": tier_of(n),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "error": deny_message(n),
        }
    try:
        text = fn(args)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        ok = bool((text or "").strip()) and not str(text).lower().startswith("error")
        return {
            "ok": ok,
            "name": n,
            "tier": tier_of(n),
            "latency_ms": ms,
            "result": (text or "")[:20000],
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("tool %s failed", n)
        return {
            "ok": False,
            "name": n,
            "tier": tier_of(n),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "error": f"{type(e).__name__}: {e}",
        }
