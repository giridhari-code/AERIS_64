"""Project context — Claude Code–similar workspace awareness (read-only)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sentinel.zeroz.workspace import default_root, list_dir

_CONTEXT_FILES = (
    "README.md",
    "README.rst",
    "AGENTS.md",
    "CLAUDE.md",
    "package.json",
    "pyproject.toml",
    "configs/zeroz.yaml",
)


def find_project_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path(os.getcwd())).resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
        if len(p.parts) <= 2:
            break
    return cur


def load_project_context(root: Optional[Path] = None, max_chars: int = 6000) -> str:
    root = (root or find_project_root()).resolve()
    parts = [f"Project root: {root}"]
    # top-level listing
    try:
        parts.append("Top-level entries:\n" + list_dir(".", root=root, max_entries=40))
    except Exception as e:
        parts.append(f"(list_dir error: {e})")
    used = 0
    for name in _CONTEXT_FILES:
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunk = text[:1500]
        block = f"\n--- {name} ---\n{chunk}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)
