"""MCP-facing list and dispatch for SENTINEL tools.

The catalog is the source of truth for schemas. Dispatch maps tool names to
callables. Optional tools (image generation, computer use) are only listed
when their capability gate reports available.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from neurofield.tools.catalog import tool_schemas
from neurofield.tools.limits import enforce_tool_limit, ToolLimitError
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
    from neurofield.tools.kalofacts_tools import (
        kalofact_create, kalofact_get, kalofact_list, kalofact_edit,
        kalofact_export, kalofact_share, kalofact_delete,
    )
    from neurofield.tools.ajeebwork_tools import (
        ajeebwork_file, ajeebwork_plan, ajeebwork_subagents, ajeebwork_task,
        ajeebwork_memory, ajeebwork_document, ajeebwork_draft_edit, ajeebwork_skill,
    )

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
        "ajeebwork_file": lambda action="read", path="", content="", old="", new="", allow_delete=False: ajeebwork_file(
            action, path=path, content=content, old=old, new=new, allow_delete=bool(allow_delete)
        ),
        "ajeebwork_plan": lambda objective="", steps_json="[]": ajeebwork_plan(objective, steps_json),
        "ajeebwork_subagents": lambda tasks_json="[]": ajeebwork_subagents(tasks_json),
        "ajeebwork_task": lambda action="list", name="", task_id="", schedule="": ajeebwork_task(
            action, name=name, task_id=task_id, schedule=schedule
        ),
        "ajeebwork_memory": lambda action="report", session_id="default", key="", value="", memory_enabled=True: ajeebwork_memory(
            action, session_id=session_id, key=key, value=value, memory_enabled=bool(memory_enabled)
        ),
        "ajeebwork_document": lambda kind="md", path="", title="Document", body="", headers_json="[]", rows_json="[]", slides_json="[]": ajeebwork_document(
            kind, path, title=title, body=body, headers_json=headers_json, rows_json=rows_json, slides_json=slides_json
        ),
        "ajeebwork_draft_edit": lambda path="", selected_text="", replacement="": ajeebwork_draft_edit(
            path, selected_text, replacement
        ),
        "ajeebwork_skill": lambda action="list", name="", instructions="": ajeebwork_skill(
            action, name=name, instructions=instructions
        ),
        "kalofact_create": lambda title="", content="", kind="", session_id="default": kalofact_create(
            title, content, kind=kind, session_id=session_id
        ),
        "kalofact_get": lambda artifact_id="", view="": kalofact_get(artifact_id, view=view),
        "kalofact_list": lambda session_id="", limit=20: kalofact_list(session_id, limit=int(limit)),
        "kalofact_edit": lambda artifact_id="", action="span", selected="", replacement="", content="", version=0, view_mode="", state_key="", state_value="": kalofact_edit(
            artifact_id, action=action, selected=selected, replacement=replacement,
            content=content, version=int(version or 0), view_mode=view_mode,
            state_key=state_key, state_value=state_value,
        ),
        "kalofact_export": lambda artifact_id="", dest_path="": kalofact_export(artifact_id, dest_path),
        "kalofact_share": lambda artifact_id="", token="", action="create": kalofact_share(
            artifact_id=artifact_id, token=token, action=action
        ),
    }

    if image_generation_available():
        table["image_generation"] = lambda prompt="", size="1024x1024": image_generation(
            prompt, size=size
        )
    if computer_use_available():
        table["computer_use"] = lambda action="", **kw: computer_use(action, **kw)

    return table


def active_tool_schemas(*, strict_limit: bool = True) -> list[dict[str, Any]]:
    """Schemas for tools that are actually dispatchable right now.

    The returned list is capped by ``max_tools_per_request``. If the active
    set is larger and ``strict_limit`` is True, ``ToolLimitError`` is raised
    so callers switch to ranked selection via tool_search.
    """
    dispatch = _build_dispatch()
    schemas = [s for s in tool_schemas() if s.get("name") in dispatch]
    return enforce_tool_limit(schemas, strict=strict_limit)


def mcp_list_tools(*, strict_limit: bool = True) -> dict[str, Any]:
    try:
        tools = active_tool_schemas(strict_limit=strict_limit)
        return {"tools": tools, "count": len(tools)}
    except ToolLimitError as e:
        return {
            "tools": [],
            "count": 0,
            "error": str(e),
            "hint": "Call tools_for_request(query) or tool_search to rank a subset.",
        }


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
