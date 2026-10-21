"""0z workspace — safe read-only filesystem helpers (no source writes)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from sentinel.zeroz.policy import is_source_path

# Max bytes for a single read
_MAX_READ = 200_000


def default_root() -> Path:
    return Path(os.getcwd()).resolve()


def list_dir(path: str | Path | None = None, *, root: Path | None = None, max_entries: int = 200) -> str:
    root = (root or default_root()).resolve()
    target = (root / (path or ".")).resolve() if path else root
    if not str(target).startswith(str(root)):
        return "0z: path escapes workspace root"
    if not target.exists():
        return f"0z: not found: {target}"
    if not target.is_dir():
        return f"0z: not a directory: {target}"
    lines = []
    for i, child in enumerate(sorted(target.iterdir())):
        if i >= max_entries:
            lines.append("...")
            break
        kind = "dir" if child.is_dir() else "file"
        lines.append(f"{kind}\t{child.name}")
    return "\n".join(lines) if lines else "(empty)"


def read_text_file(path: str, *, root: Path | None = None) -> str:
    root = (root or default_root()).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)):
        return "0z: path escapes workspace root"
    if is_source_path(str(target)):
        # Allow *reading* source for review in prose workflows, but label it
        pass
    if not target.is_file():
        return f"0z: not a file: {path}"
    data = target.read_bytes()[:_MAX_READ]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"0z: binary or non-utf8 file ({len(data)} bytes shown cap)"
