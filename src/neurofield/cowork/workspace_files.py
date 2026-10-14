"""Direct local file access under configured workspace roots.

Security:
  - Paths must resolve under an allowed root (symlink escape rejected).
  - Deletes require allow_delete=True on the call (and config may still deny).
  - Read/write size capped by CoworkConfig.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from neurofield.cowork.config import load_cowork_config, resolve_workspace_roots


class WorkspaceError(ValueError):
    pass


def _under_roots(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def resolve_path(path: str, roots: Optional[list[Path]] = None) -> Path:
    roots = roots or resolve_workspace_roots()
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        # Prefer first root as cwd-equivalent
        candidate = (roots[0] / raw).resolve()
    else:
        candidate = raw.resolve()
    if not _under_roots(candidate, roots):
        raise WorkspaceError(f"path outside workspace roots: {path}")
    return candidate


def read_file(path: str, max_bytes: Optional[int] = None) -> str:
    cfg = load_cowork_config()
    if not cfg.enabled:
        return "WorkspaceFiles: cowork disabled."
    limit = int(max_bytes if max_bytes is not None else cfg.max_read_bytes)
    try:
        p = resolve_path(path)
    except WorkspaceError as e:
        return f"WorkspaceFiles: {e}"
    if not p.is_file():
        return f"WorkspaceFiles: not a file: {path}"
    try:
        data = p.read_bytes()
    except OSError as e:
        return f"WorkspaceFiles read error: {e}"
    if len(data) > limit:
        data = data[:limit]
        truncated = True
    else:
        truncated = False
    text = data.decode("utf-8", errors="replace")
    suffix = "\n... [truncated]" if truncated else ""
    return f"WorkspaceFiles read ({p}):\n{text}{suffix}"


def write_file(path: str, content: str, *, create_parents: bool = True) -> str:
    cfg = load_cowork_config()
    if not cfg.enabled:
        return "WorkspaceFiles: cowork disabled."
    content = content if content is not None else ""
    if len(content.encode("utf-8")) > cfg.max_write_bytes:
        return (
            f"WorkspaceFiles: content exceeds max_write_bytes "
            f"({cfg.max_write_bytes})."
        )
    try:
        p = resolve_path(path)
    except WorkspaceError as e:
        return f"WorkspaceFiles: {e}"
    try:
        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"WorkspaceFiles write error: {e}"
    return f"WorkspaceFiles wrote {len(content)} chars → {p}"


def edit_file(path: str, old: str, new: str, *, replace_all: bool = False) -> str:
    cfg = load_cowork_config()
    if not cfg.enabled:
        return "WorkspaceFiles: cowork disabled."
    try:
        p = resolve_path(path)
    except WorkspaceError as e:
        return f"WorkspaceFiles: {e}"
    if not p.is_file():
        return f"WorkspaceFiles: not a file: {path}"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        return f"WorkspaceFiles read error: {e}"
    if old not in text:
        return "WorkspaceFiles: old string not found (no change)."
    if replace_all:
        updated = text.replace(old, new)
        count = text.count(old)
    else:
        updated = text.replace(old, new, 1)
        count = 1
    if len(updated.encode("utf-8")) > cfg.max_write_bytes:
        return "WorkspaceFiles: result would exceed max_write_bytes."
    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as e:
        return f"WorkspaceFiles write error: {e}"
    return f"WorkspaceFiles edited {p} ({count} replacement(s))."


def delete_file(path: str, *, allow_delete: bool = False) -> str:
    """Delete a file. Requires explicit allow_delete=True on every call.

    Additionally, NEUROFIELD_COWORK_ALLOW_DELETE must be enabled for the
    process, so a single mistaken tool call cannot wipe files.
    """
    cfg = load_cowork_config()
    if not cfg.enabled:
        return "WorkspaceFiles: cowork disabled."
    if not allow_delete:
        return (
            "WorkspaceFiles: deletion blocked. "
            "Pass allow_delete=true after explicit user permission."
        )
    if not cfg.allow_delete_default:
        return (
            "WorkspaceFiles: deletion blocked by policy. "
            "Set NEUROFIELD_COWORK_ALLOW_DELETE=1 to permit deletes."
        )
    try:
        p = resolve_path(path)
    except WorkspaceError as e:
        return f"WorkspaceFiles: {e}"
    if not p.exists():
        return f"WorkspaceFiles: path does not exist: {path}"
    if p.is_dir():
        return "WorkspaceFiles: refusing to delete directories (files only)."
    try:
        p.unlink()
    except OSError as e:
        return f"WorkspaceFiles delete error: {e}"
    return f"WorkspaceFiles deleted {p}"


def batch_process(
    paths: Iterable[str],
    operation: str,
    *,
    content: str = "",
    old: str = "",
    new: str = "",
    allow_delete: bool = False,
) -> str:
    cfg = load_cowork_config()
    if not cfg.enabled:
        return "WorkspaceFiles: cowork disabled."
    path_list = list(paths)
    if len(path_list) > cfg.max_batch_files:
        return (
            f"WorkspaceFiles: batch size {len(path_list)} exceeds "
            f"max_batch_files={cfg.max_batch_files}."
        )
    op = (operation or "").strip().lower()
    results: list[str] = []
    for path in path_list:
        if op == "read":
            results.append(read_file(path))
        elif op == "write":
            results.append(write_file(path, content))
        elif op == "edit":
            results.append(edit_file(path, old, new))
        elif op == "delete":
            results.append(delete_file(path, allow_delete=allow_delete))
        else:
            return f"WorkspaceFiles: unknown batch operation '{operation}'."
    return "WorkspaceFiles batch:\n" + "\n---\n".join(results)
