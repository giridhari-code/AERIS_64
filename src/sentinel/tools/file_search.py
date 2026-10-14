"""Repository-scoped file name and content search.

Only paths under the project root and an allow-listed set of directories
are searchable. Symlink escape outside the root is rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

from sentinel.tools.config import load_tools_config

_ALLOWED_ROOTS = ("src", "docs", "configs", "scripts", "tests", "data", "apps", "deploy")
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "artifacts",
    "runs",
    "checkpoints",
}
_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".sh",
    ".cfg",
    ".ini",
    ".rst",
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


def _safe_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def file_search(
    query: str,
    path: str = "",
    max_results: int | None = None,
) -> str:
    """
    Search file names and text content under allowed project directories.

    Parameters
    ----------
    query : str
        Case-insensitive substring to match.
    path : str
        Optional subdirectory relative to the repository root.
    max_results : int, optional
        Cap on reported hits (defaults from ToolsConfig).
    """
    cfg = load_tools_config()
    max_results = int(max_results if max_results is not None else cfg.file_search_max_results)

    q = (query or "").strip()
    if not q:
        return "FileSearch: empty query."
    if len(q) > 120:
        return "FileSearch: query too long (max 120 chars)."

    root = _repo_root()
    search_roots: list[Path] = []

    if path:
        candidate = (root / path).resolve()
        if not _safe_under_root(candidate, root):
            return "FileSearch: path is outside the repository."
        if not candidate.exists():
            return f"FileSearch: path not found: {path}"
        search_roots = [candidate]
    else:
        for name in _ALLOWED_ROOTS:
            p = root / name
            if p.is_dir():
                search_roots.append(p)

    if not search_roots:
        return "FileSearch: no searchable directories found."

    q_lower = q.lower()
    name_hits: list[str] = []
    content_hits: list[str] = []
    budget = max(max_results * 2, max_results)

    for base in search_roots:
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            ]
            for fn in filenames:
                full = Path(dirpath) / fn
                if not _safe_under_root(full, root):
                    continue
                try:
                    rel = full.relative_to(root).as_posix()
                except ValueError:
                    continue

                if q_lower in fn.lower() or q_lower in rel.lower():
                    name_hits.append(rel)
                    if len(name_hits) >= max_results:
                        break

                if full.suffix.lower() in _TEXT_SUFFIXES:
                    try:
                        if full.stat().st_size > cfg.file_search_max_file_bytes:
                            continue
                        text = full.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    if q_lower in text.lower():
                        for line in text.splitlines():
                            if q_lower in line.lower():
                                snippet = line.strip()[: cfg.file_search_max_snippet]
                                content_hits.append(f"{rel}: {snippet}")
                                break

                if len(name_hits) + len(content_hits) >= budget:
                    break
            if len(name_hits) + len(content_hits) >= budget:
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
