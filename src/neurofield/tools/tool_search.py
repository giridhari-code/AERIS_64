"""
Claude-style Tool Search / Tool Router for NeuroField.

Instead of loading every tool definition into context (token-heavy + accuracy drop),
this module ranks tools by relevance to the user query and returns only the top-k.

Design goals (inspired by Anthropic Tool Search):
  - Keep the full catalog out of the prompt until needed
  - Rank by keyword overlap + description match + name match
  - Return both the tool names and a short ready-to-inject description block
  - Zero extra model calls (pure ranking, fast)

When you have 30–100+ tools this becomes essential.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from neurofield.tools.catalog import tool_schemas

# Simple aliases so common user words map to tool names
_ALIASES = {
    "calculate": "calculator",
    "calc": "calculator",
    "math": "calculator",
    "compute": "calculator",
    "sum": "calculator",
    "time": "time_now",
    "date": "time_now",
    "clock": "time_now",
    "search": "web_search",
    "google": "web_search",
    "news": "web_search",
    "wiki": "wikipedia",
    "currency": "currency_convert",
    "convert": "currency_convert",
    "fx": "currency_convert",
    "exchange": "currency_convert",
    "rag": "rag_retrieve",
    "docs": "rag_retrieve",
    "knowledge": "rag_retrieve",
}



# ---------------------------------------------------------------------------
# Lightweight ranking (no embeddings required)
# ---------------------------------------------------------------------------

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "with", "and", "or", "what", "how", "when", "where", "who", "which",
    "do", "does", "did", "can", "could", "should", "would", "please", "tell",
    "me", "about", "kya", "hai", "ka", "ke", "ki", "ko", "se", "mein", "par",
}


def _tokens(text: str) -> set[str]:
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9_]+", text)
    return {w for w in words if len(w) > 1 and w not in _STOP}


@dataclass
class RankedTool:
    name: str
    description: str
    score: float
    schema: dict[str, Any]


def search_tools(
    query: str,
    top_k: int = 5,
    min_score: float = 0.15,
) -> list[RankedTool]:
    """
    Rank all tools against the user query and return the best matches.

    Scoring:
      - name exact / partial match   → high weight
      - description token overlap    → medium
      - required parameter names     → small boost
    """
    q_tokens = _tokens(query)
    # expand with aliases
    extra = set()
    for t in list(q_tokens):
        if t in _ALIASES:
            extra |= _tokens(_ALIASES[t].replace("_", " "))
            extra.add(_ALIASES[t])
    q_tokens |= extra
    if not q_tokens:
        return []

    ranked: list[RankedTool] = []
    for schema in tool_schemas():
        name = schema.get("name", "")
        desc = schema.get("description", "")
        props = (schema.get("inputSchema") or {}).get("properties") or {}

        name_l = name.lower().replace("_", " ")
        name_tokens = _tokens(name_l)
        desc_tokens = _tokens(desc)
        param_tokens = _tokens(" ".join(props.keys()))

        # Jaccard-like overlap
        name_overlap = len(q_tokens & name_tokens) / max(1, len(name_tokens))
        desc_overlap = len(q_tokens & desc_tokens) / max(1, len(q_tokens | desc_tokens))
        param_overlap = len(q_tokens & param_tokens) / max(1, len(param_tokens) or 1)

        # Exact / substring name hits are very strong
        exact = 1.0 if name_l in query.lower() else 0.0
        # any query token is a prefix/substring of the tool name (calc → calculator)
        partial = 0.0
        for t in q_tokens:
            if t in name_l or name_l.startswith(t) or any(n.startswith(t) for n in name_tokens):
                partial = max(partial, 0.85)
            elif any(t.startswith(n) or n.startswith(t) for n in name_tokens if len(n) > 3):
                partial = max(partial, 0.55)

        score = (
            2.5 * exact
            + 1.9 * partial
            + 1.4 * name_overlap
            + 1.0 * desc_overlap
            + 0.4 * param_overlap
        )

        if score >= min_score:
            ranked.append(
                RankedTool(
                    name=name,
                    description=desc,
                    score=round(score, 4),
                    schema=schema,
                )
            )

    ranked.sort(key=lambda t: t.score, reverse=True)
    return ranked[: max(1, top_k)]


def tool_search_report(query: str, top_k: int = 5) -> str:
    """
    Human-readable block that can be injected into the prompt
    (same style as other *report functions).
    """
    hits = search_tools(query, top_k=top_k)
    if not hits:
        return "ToolSearch: no relevant tools found for this query."

    lines = ["ToolSearch — most relevant tools:"]
    for i, t in enumerate(hits, 1):
        lines.append(f"  {i}. {t.name}  (score={t.score:.2f})")
        lines.append(f"     {t.description}")
    return "\n".join(lines)


def get_relevant_schemas(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return only the JSON schemas of the top-k tools (for agent / MCP use)."""
    return [t.schema for t in search_tools(query, top_k=top_k)]


def format_tool_catalog_for_prompt(
    query: str,
    top_k: int = 6,
    max_chars: int = 1800,
) -> str:
    """
    Compact catalog block to put in the system / tool section of a prompt.
    Only the relevant tools are shown → saves tokens (Claude-style).
    """
    hits = search_tools(query, top_k=top_k)
    if not hits:
        return ""

    parts = ["Available tools (ranked for this query):"]
    for t in hits:
        line = f"- {t.name}: {t.description}"
        parts.append(line)

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n... (truncated)"
    return text
