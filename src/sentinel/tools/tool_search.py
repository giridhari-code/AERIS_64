"""Tool router: rank catalog tools against a natural-language need.

Used to keep large tool sets out of the prompt until needed. Ranking is
deterministic (token overlap + alias expansion). Only tools that are currently
dispatchable are considered when ``tool_search_only_active`` is true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from sentinel.tools.catalog import tool_schemas
from sentinel.tools.config import load_tools_config
from sentinel.tools.limits import enforce_tool_limit, max_tools_per_request

# Query tokens mapped to canonical tool names (stable, reviewed vocabulary).
_ALIASES: dict[str, str] = {
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
    "code": "code_execution",
    "python": "code_execution",
    "interpreter": "code_execution",
    "bash": "bash",
    "shell": "bash",
    "terminal": "bash",
    "fetch": "web_fetch",
    "url": "web_fetch",
    "file": "file_search",
    "find": "file_search",
    "image": "image_generation",
    "screenshot": "computer_use",
    "click": "computer_use",
}

_STOP = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
        "for", "with", "and", "or", "what", "how", "when", "where", "who", "which",
        "do", "does", "did", "can", "could", "should", "would", "please", "tell",
        "me", "about", "kya", "hai", "ka", "ke", "ki", "ko", "se", "mein", "par",
        "this", "that", "need", "want", "use", "using", "tool", "tools",
    }
)

# Ranking weights (documented; not magic without comment)
_W_EXACT_NAME = 3.0
_W_ALIAS_HIT = 2.4
_W_PARTIAL_NAME = 1.6
_W_NAME_OVERLAP = 1.3
_W_DESC_OVERLAP = 1.0
_W_PARAM_OVERLAP = 0.35


@dataclass(frozen=True)
class RankedTool:
    name: str
    description: str
    score: float
    schema: dict[str, Any]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return {w for w in words if len(w) > 1 and w not in _STOP}


def _expand_aliases(tokens: set[str]) -> set[str]:
    extra: set[str] = set()
    for t in tokens:
        target = _ALIASES.get(t)
        if not target:
            continue
        extra.add(target)
        extra |= _tokens(target.replace("_", " "))
    return tokens | extra


def _active_names() -> Optional[set[str]]:
    """Names that mcp_support can dispatch right now; None = do not filter."""
    cfg = load_tools_config()
    if not cfg.tool_search_only_active:
        return None
    try:
        from sentinel.tools.mcp_support import active_tool_schemas

        return {s["name"] for s in active_tool_schemas() if "name" in s}
    except Exception:
        return None


def _candidate_schemas() -> list[dict[str, Any]]:
    active = _active_names()
    schemas = tool_schemas()
    if active is None:
        return list(schemas)
    return [s for s in schemas if s.get("name") in active]


def _score_schema(query: str, q_tokens: set[str], schema: dict[str, Any]) -> float:
    name = str(schema.get("name") or "")
    desc = str(schema.get("description") or "")
    props = (schema.get("inputSchema") or {}).get("properties") or {}

    name_l = name.lower().replace("_", " ")
    name_tokens = _tokens(name_l)
    desc_tokens = _tokens(desc)
    param_tokens = _tokens(" ".join(props.keys()))

    name_overlap = len(q_tokens & name_tokens) / max(1, len(name_tokens))
    union_desc = q_tokens | desc_tokens
    desc_overlap = len(q_tokens & desc_tokens) / max(1, len(union_desc))
    param_overlap = (
        len(q_tokens & param_tokens) / max(1, len(param_tokens)) if param_tokens else 0.0
    )

    exact = 1.0 if name_l in query.lower() or name.lower() in query.lower() else 0.0

    alias_hit = 0.0
    for t in q_tokens:
        if _ALIASES.get(t) == name or t == name:
            alias_hit = 1.0
            break

    partial = 0.0
    for t in q_tokens:
        if t in name_l or name_l.startswith(t):
            partial = max(partial, 1.0)
        elif any(n.startswith(t) or t.startswith(n) for n in name_tokens if len(n) > 2):
            partial = max(partial, 0.7)

    return (
        _W_EXACT_NAME * exact
        + _W_ALIAS_HIT * alias_hit
        + _W_PARTIAL_NAME * partial
        + _W_NAME_OVERLAP * name_overlap
        + _W_DESC_OVERLAP * desc_overlap
        + _W_PARAM_OVERLAP * param_overlap
    )


def search_tools(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[RankedTool]:
    """
    Rank active tools for ``query`` and return the best matches.

    Empty query or no matches above ``min_score`` yields an empty list.
    """
    cfg = load_tools_config()
    top_k = int(top_k if top_k is not None else cfg.tool_search_top_k)
    min_score = float(min_score if min_score is not None else cfg.tool_search_min_score)
    top_k = max(1, top_k)

    q = (query or "").strip()
    if not q:
        return []

    q_tokens = _expand_aliases(_tokens(q))
    if not q_tokens:
        return []

    ranked: list[RankedTool] = []
    for schema in _candidate_schemas():
        name = schema.get("name")
        if not name:
            continue
        score = _score_schema(q, q_tokens, schema)
        if score < min_score:
            continue
        ranked.append(
            RankedTool(
                name=str(name),
                description=str(schema.get("description") or ""),
                score=round(score, 4),
                schema=schema,
            )
        )

    ranked.sort(key=lambda t: t.score, reverse=True)
    return ranked[:top_k]


def tool_search_report(query: str, top_k: int | None = None) -> str:
    """Prompt-injectable ranking summary."""
    hits = search_tools(query, top_k=top_k)
    if not hits:
        return "ToolSearch: no relevant tools above threshold."

    lines = ["ToolSearch — ranked tools:"]
    for i, t in enumerate(hits, 1):
        lines.append(f"  {i}. {t.name}  (score={t.score:.2f})")
        if t.description:
            lines.append(f"     {t.description}")
    return "\n".join(lines)


def get_relevant_schemas(
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Schemas for the top-ranked tools (agent / MCP)."""
    return [t.schema for t in search_tools(query, top_k=top_k)]


def route_tool_names(query: str, top_k: int | None = None) -> list[str]:
    """Ordered tool names for an executor to try."""
    return [t.name for t in search_tools(query, top_k=top_k)]


def format_tool_catalog_for_prompt(
    query: str,
    top_k: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Compact ranked catalog for system / tool context (token-bounded)."""
    cfg = load_tools_config()
    max_chars = int(
        max_chars if max_chars is not None else cfg.tool_search_catalog_max_chars
    )
    hits = search_tools(query, top_k=top_k)
    if not hits:
        return ""

    parts = ["Available tools (ranked for this query):"]
    for t in hits:
        parts.append(f"- {t.name}: {t.description}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n... [truncated]"
    return text


def tools_for_request(
    query: str | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Schemas safe to attach to a single model/agent turn.

    - With ``query``: ranked subset via ``search_tools`` (recommended at scale).
    - Without ``query``: full active catalog, still capped by
      ``max_tools_per_request`` (non-strict truncate if over limit).
    """
    if query and query.strip():
        return get_relevant_schemas(query, top_k=top_k)
    # Unscoped list: prefer active dispatchable tools, hard-cap for the request.
    try:
        from sentinel.tools.mcp_support import active_tool_schemas
        schemas = active_tool_schemas(strict_limit=False)
    except Exception:
        schemas = list(tool_schemas())
    return enforce_tool_limit(schemas, strict=False)

