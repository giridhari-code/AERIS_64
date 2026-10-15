"""Supported Kalofact content kinds."""

from __future__ import annotations

from enum import Enum


class ArtifactKind(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    REACT = "react"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    BASH = "bash"
    MERMAID = "mermaid"
    SVG = "svg"
    JSON = "json"
    TEXT = "text"
    UNKNOWN = "unknown"


# Preview-capable kinds (code/source always available)
PREVIEW_KINDS = frozenset(
    {
        ArtifactKind.HTML,
        ArtifactKind.REACT,
        ArtifactKind.MERMAID,
        ArtifactKind.SVG,
        ArtifactKind.MARKDOWN,
    }
)


def detect_kind(content: str, hint: str | None = None) -> ArtifactKind:
    if hint:
        h = hint.lower().strip()
        for k in ArtifactKind:
            if k.value == h or k.name.lower() == h:
                return k
    c = (content or "").lstrip()
    if c.startswith("```mermaid") or "graph " in c[:80] or "sequenceDiagram" in c[:120]:
        return ArtifactKind.MERMAID
    if c.startswith("<svg") or c.startswith("<?xml"):
        return ArtifactKind.SVG
    if c.startswith("<!DOCTYPE") or c.startswith("<html") or c.startswith("<div"):
        return ArtifactKind.HTML
    if "import React" in c or "export default function" in c:
        return ArtifactKind.REACT
    if c.startswith("```python") or c.startswith("def ") or c.startswith("import "):
        return ArtifactKind.PYTHON
    if c.startswith("SELECT ") or c.startswith("WITH "):
        return ArtifactKind.SQL
    if c.startswith("#") or "## " in c[:200]:
        return ArtifactKind.MARKDOWN
    return ArtifactKind.TEXT
