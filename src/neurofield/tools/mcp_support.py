"""Minimal MCP (Model Context Protocol) helpers.

MCP is a protocol for exposing tools/resources to models. NeuroField already
exposes an MCP-style catalog via ``tool_schemas()``. This module adds thin
helpers so an MCP server or client can list tools and dispatch by name.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from neurofield.tools.catalog import tool_schemas, list_mcp_tools


# Dispatch map — filled lazily to avoid circular imports at module load
_DISPATCH: dict[str, Callable[..., str]] = {}


def _ensure_dispatch() -> dict[str, Callable[..., str]]:
    if _DISPATCH:
        return _DISPATCH
    # Local imports
    from neurofield.tools.calculator import calculate
    from neurofield.tools.currency import convert_currency
    from neurofield.tools.timeutil import now_report
    from neurofield.tools.weather import weather_report
    from neurofield.tools.web_search import web_search_report
    from neurofield.tools.wikipedia import wikipedia_summary
    from neurofield.tools.git_tool import git_report
    from neurofield.tools.rag_tool import rag_retrieve
    from neurofield.tools.tool_search import tool_search_report
    from neurofield.tools.code_execution import code_execution
    from neurofield.tools.bash_tool import bash
    from neurofield.tools.web_fetch import web_fetch
    from neurofield.tools.file_search import file_search
    from neurofield.tools.image_generation import image_generation
    from neurofield.tools.computer_use import computer_use

    _DISPATCH.update(
        {
            "calculator": lambda expression="": calculate(expression),
            "currency_convert": lambda amount=0, **kw: convert_currency(
                float(amount), kw.get("from", "USD"), kw.get("to", "INR")
            ),
            "time_now": lambda timezone=None: now_report(timezone),
            "weather": lambda location="Delhi": weather_report(location),
            "web_search": lambda query="", max_results=5: web_search_report(
                query, max_results=int(max_results)
            ),
            "wikipedia": lambda topic="", lang="en": wikipedia_summary(topic, lang=lang)
            if "lang" in wikipedia_summary.__code__.co_varnames
            else wikipedia_summary(topic),
            "git": lambda command="status", message="", path="": git_report(
                command, message=message or None, path=path or None
            ),
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
            "file_search": lambda query="", path="": file_search(query, path=path),
            "image_generation": lambda prompt="", size="1024x1024": image_generation(
                prompt, size=size
            ),
            "computer_use": lambda action="", **kw: computer_use(action, **kw),
        }
    )
    return _DISPATCH


def mcp_list_tools() -> dict[str, Any]:
    """Return the MCP tools list payload."""
    return list_mcp_tools()


def mcp_call_tool(name: str, arguments: Optional[dict[str, Any]] = None) -> str:
    """
    Dispatch a tool by name with a dict of arguments.
    Returns the tool's string result (or an error message).
    """
    arguments = arguments or {}
    dispatch = _ensure_dispatch()
    fn = dispatch.get(name)
    if fn is None:
        known = ", ".join(sorted(dispatch.keys()))
        return f"MCP: unknown tool '{name}'. Known: {known}"
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"MCP: bad arguments for '{name}': {e}"
    except Exception as e:
        return f"MCP: tool '{name}' failed: {e}"
