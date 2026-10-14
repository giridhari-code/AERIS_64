"""File Search — find files and content inside the repository (safe paths only)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


# Only search under these roots relative to repo / cwd
_ALLOWED_ROOTS = ("src", "docs", "configs", "scripts", "tests", "data", "apps")
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", "artifacts"}
_MAX_HITS = 12
_MAX_SNIPPET = 180


def _repo_root() -> Path:
    # Prefer walking up from this file
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def file_search(
    query: str,
    path: str = "",
    max_results: int = 8,
) -> str:
    """
    Search file names and (lightly) file contents under allowed directories.
    """
    query = (query or "").strip()
    if not query:
        return "FileSearch: empty query."
    if len(query) > 120:
        return "FileSearch: query too long."

    root = _repo_root()
    search_roots: list[Path] = []

    if path:
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return "FileSearch: path outside repository is not allowed."
        if candidate.exists():
            search_roots = [candidate]
        else:
            return f"FileSearch: path not found: {path}"
    else:
        for name in _ALLOWED_ROOTS:
            p = root / name
            if p.is_dir():
                search_roots.append(p)

    if not search_roots:
        search_roots = [root]

    q_lower = query.lower()
    name_hits: list[str] = []
    content_hits: list[str] = []

    for base in search_roots:
        for dirpath, dirnames, filenames in os.walk(base):
            # prune
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = full.relative_to(root).as_posix()
                # name match
                if q_lower in fn.lower() or q_lower in rel.lower():
                    name_hits.append(rel)
                    if len(name_hits) >= max_results:
                        break
                # light content match for text-ish files
                if full.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh"}:
                    try:
                        if full.stat().st_size > 400_000:
                            continue
                        text = full.read_text(encoding="utf-8", errors="ignore")
                        if q_lower in text.lower():
                            # first matching line
                            for line in text.splitlines():
                                if q_lower in line.lower():
                                    snippet = line.strip()[:_MAX_SNIPPET]
                                    content_hits.append(f"{rel}: {snippet}")
                                    break
                    except Exception:
                        pass
                if len(name_hits) + len(content_hits) >= _MAX_HITS:
                    break
            if len(name_hits) + len(content_hits) >= _MAX_HITS:
                break

    lines = ["FileSearch results:"]
    if name_hits:
        lines.append("  [filename matches]")
        for h in name_hits[:max_results]:
            lines.append(f"  - {h}")
    if content_hits:
        lines.append("  [content matches]")
        for h in content_hits[:max_results]:
            lines.append(f"  - {h}")
    if len(lines) == 1:
        lines.append("  (no matches)")
    return "\n".join(lines)
