"""Precision inline editing and versioned updates for Kalofacts."""

from __future__ import annotations

import time
from typing import Optional

from neurofield.kalofacts.artifact import Kalofact
from neurofield.kalofacts.config import load_kalofacts_config
from neurofield.kalofacts.store import get, save


def set_view_mode(artifact_id: str, mode: str) -> str:
    mode = (mode or "").lower().strip()
    if mode not in {"preview", "code"}:
        return "Kalofacts: view_mode must be 'preview' or 'code'."
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    if mode == "preview" and not art.supports_preview():
        return f"Kalofacts: kind '{art.kind}' has no live preview; use code view."
    art.view_mode = mode
    art.updated_at = time.time()
    save(art)
    return f"Kalofacts {artifact_id} view_mode={mode}"


def replace_span(
    artifact_id: str,
    selected: str,
    replacement: str,
    *,
    replace_all: bool = False,
    note: str = "inline edit",
) -> str:
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    if not selected:
        return "Kalofacts: selected text is required."
    if selected not in art.content:
        return "Kalofacts: selected span not found (no change)."
    cfg = load_kalofacts_config()
    if replace_all:
        new_content = art.content.replace(selected, replacement)
        count = art.content.count(selected)
    else:
        new_content = art.content.replace(selected, replacement, 1)
        count = 1
    if len(new_content.encode("utf-8")) > cfg.max_content_bytes:
        return "Kalofacts: result exceeds max_content_bytes."
    return _commit_version(art, new_content, note=f"{note} ({count} span(s))")


def replace_all_content(
    artifact_id: str,
    content: str,
    *,
    note: str = "full update",
) -> str:
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    cfg = load_kalofacts_config()
    content = content or ""
    if len(content.encode("utf-8")) > cfg.max_content_bytes:
        return "Kalofacts: content exceeds max_content_bytes."
    return _commit_version(art, content, note=note)


def _commit_version(art: Kalofact, content: str, *, note: str) -> str:
    cfg = load_kalofacts_config()
    versions = list(art.versions or [])
    next_v = (versions[-1]["version"] + 1) if versions else 1
    versions.append(
        {
            "version": next_v,
            "content": content,
            "created_at": time.time(),
            "note": note,
        }
    )
    # Bound history
    if len(versions) > cfg.max_versions:
        versions = versions[-cfg.max_versions :]
    art.versions = versions
    art.content = content
    art.updated_at = time.time()
    save(art)
    return f"Kalofacts {art.artifact_id} updated to v{next_v} ({note})"


def restore_version(artifact_id: str, version: int) -> str:
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    match = next((v for v in art.versions if int(v.get("version", -1)) == int(version)), None)
    if not match:
        return f"Kalofacts: version {version} not found."
    return _commit_version(
        art, str(match.get("content") or ""), note=f"restore v{version}"
    )


def set_state(artifact_id: str, key: str, value: str) -> str:
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    state = dict(art.state or {})
    state[str(key)[:128]] = value[:4000]
    art.state = state
    art.updated_at = time.time()
    save(art)
    return f"Kalofacts state set {key}"
