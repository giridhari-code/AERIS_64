"""Tool adapters for Kalofacts."""

from __future__ import annotations

from typing import Optional

from sentinel.kalofacts import (
    create_and_save,
    get,
    list_artifacts,
    delete,
    should_auto_open,
    set_view_mode,
    replace_span,
    replace_all_content,
    restore_version,
    set_state,
    export_source,
    create_share_token,
    resolve_share,
)


def kalofact_create(
    title: str,
    content: str,
    kind: str = "",
    session_id: str = "default",
) -> str:
    try:
        art = create_and_save(
            title, content, kind=kind or None, session_id=session_id
        )
    except Exception as e:
        return f"Kalofacts create error: {type(e).__name__}: {e}"
    return (
        f"Kalofact created id={art.artifact_id} kind={art.kind} "
        f"view={art.view_mode} title={art.title}"
    )


def kalofact_get(artifact_id: str, view: str = "") -> str:
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    mode = (view or art.view_mode).lower()
    header = (
        f"Kalofact id={art.artifact_id} title={art.title} kind={art.kind} "
        f"view={mode} versions={len(art.versions)}"
    )
    if mode == "code":
        return f"{header}\n--- code ---\n{art.content}"
    return f"{header}\n--- preview ({art.kind}) ---\n{art.content}"


def kalofact_list(session_id: str = "", limit: int = 20) -> str:
    items = list_artifacts(session_id=session_id or None, limit=int(limit))
    if not items:
        return "Kalofacts: library empty."
    lines = ["Kalofacts library:"]
    for e in items:
        lines.append(
            f"  {e.get('artifact_id')} [{e.get('kind')}] {e.get('title')}"
        )
    return "\n".join(lines)


def kalofact_edit(
    artifact_id: str,
    action: str = "span",
    selected: str = "",
    replacement: str = "",
    content: str = "",
    version: int = 0,
    view_mode: str = "",
    state_key: str = "",
    state_value: str = "",
) -> str:
    action = (action or "span").lower()
    if action == "view":
        return set_view_mode(artifact_id, view_mode or "preview")
    if action == "span":
        return replace_span(artifact_id, selected, replacement)
    if action == "replace":
        return replace_all_content(artifact_id, content)
    if action == "restore":
        return restore_version(artifact_id, int(version))
    if action == "state":
        return set_state(artifact_id, state_key, state_value)
    return f"Kalofacts edit: unknown action '{action}'"


def kalofact_export(artifact_id: str, dest_path: str = "") -> str:
    return export_source(artifact_id, dest_path or None)


def kalofact_share(artifact_id: str = "", token: str = "", action: str = "create") -> str:
    action = (action or "create").lower()
    if action == "resolve":
        return resolve_share(token)
    return create_share_token(artifact_id)


def kalofact_delete(artifact_id: str) -> str:
    if not artifact_id:
        return "Kalofacts: artifact_id required."
    delete(artifact_id)
    return f"Kalofacts deleted {artifact_id}"


def kalofact_should_open(content: str) -> str:
    return "yes" if should_auto_open(content) else "no"
