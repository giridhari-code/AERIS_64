"""MCP-facing list and dispatch for NeuroField tools.

The catalog is the source of truth for schemas. Dispatch maps tool names to
callables. Optional tools (image generation, computer use) are only listed
when their capability gate reports available.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from neurofield.tools.catalog import tool_schemas
from neurofield.tools.computer_use import computer_use, computer_use_available
from neurofield.tools.image_generation import image_generation, image_generation_available


def _build_dispatch() -> dict[str, Callable[..., str]]:
    from neurofield.tools.bash_tool import bash
    from neurofield.tools.calculator import calculate
    from neurofield.tools.code_execution import code_execution
    from neurofield.tools.currency import convert_currency
    from neurofield.tools.file_search import file_search
    from neurofield.tools.git_tool import git_report
    from neurofield.tools.rag_tool import rag_retrieve
    from neurofield.tools.timeutil import now_report
    from neurofield.tools.tool_search import tool_search_report
    from neurofield.tools.weather import weather_report
    from neurofield.tools.web_fetch import web_fetch
    from neurofield.tools.web_search import web_search_report
    from neurofield.tools.wikipedia import wikipedia_summary

    table: dict[str, Callable[..., str]] = {
        "calculator": lambda expression="": calculate(expression),
        "currency_convert": lambda amount=0, **kw: convert_currency(
            float(amount), str(kw.get("from", "USD")), str(kw.get("to", "INR"))
        ),
        "time_now": lambda timezone=None: now_report(timezone),
        "weather": lambda location="Delhi": weather_report(location),
        "web_search": lambda query="", max_results=5: web_search_report(
            query, max_results=int(max_results)
        ),
        "wikipedia": lambda topic="", lang="en": wikipedia_summary(topic),
        "git": lambda command="status", message="", path="": git_report(command),
        "rag_retrieve": lambda query="", top_k=4: rag_retrieve(query, top_k=int(top_k)),
        "tool_search": lambda query="", top_k=5: tool_search_report(
            query, top_k=int(top_k)
        ),
        "code_execution": lambda code="", language="python": code_execution(
            code, language=language
        ),
        "code_interpreter": lambda code="", language="python": code_execution(
            code, language=language
        ),
        "bash": lambda command="": bash(command),
        "web_fetch": lambda url="", max_chars=8000: web_fetch(
            url, max_chars=int(max_chars)
        ),
        "file_search": lambda query="", path="": file_search(query, path=path or ""),
    }

    if image_generation_available():
        table["image_generation"] = lambda prompt="", size="1024x1024": image_generation(
            prompt, size=size
        )
    if computer_use_available():
        table["computer_use"] = lambda action="", **kw: computer_use(action, **kw)

    return table


def active_tool_schemas() -> list[dict[str, Any]]:
    """Schemas for tools that are actually dispatchable right now."""
    dispatch = _build_dispatch()
    return [s for s in tool_schemas() if s.get("name") in dispatch]


def mcp_list_tools() -> dict[str, Any]:
    return {"tools": active_tool_schemas()}


def mcp_call_tool(name: str, arguments: Optional[dict[str, Any]] = None) -> str:
    """Invoke a tool by name. Unknown or disabled tools fail closed."""
    arguments = dict(arguments or {})
    dispatch = _build_dispatch()
    fn = dispatch.get(name)
    if fn is None:
        known = ", ".join(sorted(dispatch.keys()))
        return f"MCP: tool '{name}' is unknown or disabled. Available: {known}"
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"MCP: invalid arguments for '{name}': {e}"
    except Exception as e:
        return f"MCP: '{name}' failed: {type(e).__name__}: {e}"
