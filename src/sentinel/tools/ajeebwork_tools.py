"""Tool-facing adapters for AjeebWork capabilities."""

from __future__ import annotations

import json
from typing import Any, Optional

from sentinel.ajeebwork import (
    read_file,
    write_file,
    edit_file,
    delete_file,
    batch_process,
    plan_execute_report,
    subagents_report,
    tasks_report,
    create_task,
    run_task_inline,
    memory_get,
    memory_set,
    memory_report,
    documents_report,
    edit_span,
    save_skill,
    load_skill,
    list_skills,
    call_connector,
)


def ajeebwork_file(
    action: str,
    path: str = "",
    content: str = "",
    old: str = "",
    new: str = "",
    paths: Optional[list[str]] = None,
    allow_delete: bool = False,
) -> str:
    action = (action or "read").lower().strip()
    if action == "read":
        return read_file(path)
    if action == "write":
        return write_file(path, content)
    if action == "edit":
        return edit_file(path, old, new)
    if action == "delete":
        return delete_file(path, allow_delete=allow_delete)
    if action == "batch":
        return batch_process(
            paths or [],
            content and "write" or "read",
            content=content,
            old=old,
            new=new,
            allow_delete=allow_delete,
        )
    return f"ajeebwork_file: unknown action '{action}'"


def ajeebwork_plan(objective: str, steps_json: str = "[]") -> str:
    try:
        specs = json.loads(steps_json) if steps_json else []
        if not isinstance(specs, list):
            return "ajeebwork_plan: steps_json must be a JSON list."
    except json.JSONDecodeError as e:
        return f"ajeebwork_plan: invalid JSON: {e}"
    return plan_execute_report(objective, specs, run_step=None)


def ajeebwork_subagents(tasks_json: str = "[]") -> str:
    try:
        tasks = json.loads(tasks_json) if tasks_json else []
        if not isinstance(tasks, list):
            return "ajeebwork_subagents: tasks_json must be a JSON list."
    except json.JSONDecodeError as e:
        return f"ajeebwork_subagents: invalid JSON: {e}"
    return subagents_report(tasks, worker=None)


def ajeebwork_task(
    action: str = "list",
    name: str = "",
    task_id: str = "",
    schedule: str = "",
) -> str:
    action = (action or "list").lower()
    if action == "list":
        return tasks_report()
    if action == "status":
        return tasks_report(task_id or None)
    if action == "create":
        rec = create_task(name or "task", schedule=schedule or None)
        return f"Task created {rec.task_id} status={rec.status}"
    if action == "run_inline" and name:
        rec = run_task_inline(name, lambda r: f"inline work for {r.name}")
        return tasks_report(rec.task_id)
    return f"ajeebwork_task: unknown action '{action}'"


def ajeebwork_memory(
    action: str = "report",
    session_id: str = "default",
    key: str = "",
    value: str = "",
    memory_enabled: bool = True,
) -> str:
    action = (action or "report").lower()
    if action == "get":
        return memory_get(session_id, key, memory_enabled=memory_enabled)
    if action == "set":
        return memory_set(session_id, key, value, memory_enabled=memory_enabled)
    return memory_report(session_id, memory_enabled=memory_enabled)


def ajeebwork_document(
    kind: str,
    path: str,
    title: str = "Document",
    body: str = "",
    headers_json: str = "[]",
    rows_json: str = "[]",
    slides_json: str = "[]",
) -> str:
    kwargs: dict[str, Any] = {"title": title, "body": body}
    try:
        if headers_json:
            kwargs["headers"] = json.loads(headers_json)
        if rows_json:
            kwargs["rows"] = json.loads(rows_json)
        if slides_json:
            kwargs["slides"] = json.loads(slides_json)
    except json.JSONDecodeError as e:
        return f"ajeebwork_document: invalid JSON: {e}"
    return documents_report(kind, path, **kwargs)


def ajeebwork_draft_edit(path: str, selected_text: str, replacement: str) -> str:
    return edit_span(path, selected_text, replacement)


def ajeebwork_skill(
    action: str = "list",
    name: str = "",
    instructions: str = "",
) -> str:
    action = (action or "list").lower()
    if action == "list":
        return list_skills()
    if action == "load":
        return load_skill(name)
    if action == "save":
        return save_skill(name, instructions)
    if action == "connector":
        return call_connector(name)
    return f"ajeebwork_skill: unknown action '{action}'"
