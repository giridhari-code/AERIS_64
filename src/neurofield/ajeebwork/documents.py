"""Professional document creation without mandatory heavy deps.

Produces:
  - Markdown reports (always)
  - CSV tables (stdlib)
  - TSV / plain structured text suitable for Excel import
Optional xlsx/pptx can be added by the serving layer when those packages
are installed; core stays dependency-free.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Optional, Sequence

from neurofield.ajeebwork.config import load_ajeebwork_config
from neurofield.ajeebwork.workspace_files import resolve_path, WorkspaceError, write_file


def write_markdown_report(path: str, title: str, body: str, *, sections: Optional[list[dict[str, str]]] = None) -> str:
    parts = [f"# {title}", "", body.strip(), ""]
    for sec in sections or []:
        parts.append(f"## {sec.get('heading', 'Section')}")
        parts.append("")
        parts.append(sec.get("body", ""))
        parts.append("")
    return write_file(path, "\n".join(parts))


def write_csv_table(
    path: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(headers))
    for row in rows:
        w.writerow(list(row))
    return write_file(path, buf.getvalue())


def write_excel_compatible_csv(
    path: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    formulas: Optional[dict[str, str]] = None,
) -> str:
    """
    Write CSV plus a sidecar ``.formulas.json`` description so a later xlsx
    adapter can materialize real spreadsheet formulas.
    """
    import json

    result = write_csv_table(path, headers, rows)
    if formulas:
        try:
            p = resolve_path(path)
            side = p.with_suffix(p.suffix + ".formulas.json")
            side.write_text(json.dumps(formulas, indent=2), encoding="utf-8")
            result += f"\nFormulas metadata → {side}"
        except WorkspaceError as e:
            result += f"\nFormulas metadata skipped: {e}"
    return result


def write_presentation_outline(path: str, title: str, slides: list[dict[str, Any]]) -> str:
    """Markdown slide outline (convertible to PPTX by an external adapter)."""
    lines = [f"# {title}", ""]
    for i, slide in enumerate(slides, 1):
        lines.append(f"## Slide {i}: {slide.get('title', 'Untitled')}")
        bullets = slide.get("bullets") or []
        for b in bullets:
            lines.append(f"- {b}")
        notes = slide.get("notes")
        if notes:
            lines.append(f"> Notes: {notes}")
        lines.append("")
    return write_file(path, "\n".join(lines))


def documents_report(kind: str, path: str, **kwargs: Any) -> str:
    kind = (kind or "").lower().strip()
    if kind in {"md", "markdown", "report"}:
        return write_markdown_report(
            path,
            str(kwargs.get("title") or "Report"),
            str(kwargs.get("body") or ""),
            sections=kwargs.get("sections"),
        )
    if kind in {"csv", "table", "excel_csv"}:
        return write_excel_compatible_csv(
            path,
            kwargs.get("headers") or [],
            kwargs.get("rows") or [],
            formulas=kwargs.get("formulas"),
        )
    if kind in {"pptx_outline", "slides", "presentation"}:
        return write_presentation_outline(
            path,
            str(kwargs.get("title") or "Presentation"),
            list(kwargs.get("slides") or []),
        )
    return f"Documents: unknown kind '{kind}' (md|csv|pptx_outline)."
