"""In-place draft editing for Markdown (and plain text) spans.

Caller supplies the draft path, a selected span (exact text), and the
revision instruction or replacement. No silent whole-file rewrite.
"""

from __future__ import annotations

from sentinel.ajeebwork.workspace_files import edit_file, read_file, resolve_path, WorkspaceError


def edit_span(
    path: str,
    selected_text: str,
    replacement: str,
    *,
    replace_all: bool = False,
) -> str:
    selected_text = selected_text or ""
    replacement = replacement if replacement is not None else ""
    if not selected_text:
        return "DraftEdit: selected_text is required."
    return edit_file(path, selected_text, replacement, replace_all=replace_all)


def preview_span(path: str, selected_text: str, context_chars: int = 80) -> str:
    try:
        p = resolve_path(path)
        text = p.read_text(encoding="utf-8")
    except WorkspaceError as e:
        return f"DraftEdit: {e}"
    except OSError as e:
        return f"DraftEdit read error: {e}"
    idx = text.find(selected_text or "")
    if idx < 0:
        return "DraftEdit: selected_text not found in draft."
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(selected_text) + context_chars)
    return f"DraftEdit preview ({p}):\n{text[start:end]}"
