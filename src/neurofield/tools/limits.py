"""Per-request tool surface limits.

Mirrors frontier API practice (e.g. xAI ~200 custom tools per request):
the installed catalog may be large, but one model/agent turn must not be
handed more than ``max_tools_per_request`` schemas.

Enforcement is fail-closed when the caller asks for an uncapped full list
that exceeds the limit; prefer ``tool_search`` / ``tools_for_request`` to
select a ranked subset.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from neurofield.tools.config import load_tools_config


class ToolLimitError(ValueError):
    """Raised when a request would expose more tools than allowed."""


def max_tools_per_request() -> int:
    return max(1, int(load_tools_config().max_tools_per_request))


def enforce_tool_limit(
    schemas: Sequence[dict[str, Any]],
    *,
    limit: Optional[int] = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """
    Cap or reject a schema list for one request.

    Parameters
    ----------
    schemas :
        Candidate tool schemas for this turn.
    limit :
        Override for ``ToolsConfig.max_tools_per_request``.
    strict :
        If True and len(schemas) > limit, raise ``ToolLimitError``.
        If False, silently truncate to ``limit`` (stable order preserved).
    """
    cap = max(1, int(limit if limit is not None else max_tools_per_request()))
    items = list(schemas)
    if len(items) <= cap:
        return items
    if strict:
        raise ToolLimitError(
            f"Tool limit exceeded: {len(items)} schemas requested, "
            f"max_tools_per_request={cap}. "
            "Use tool_search / tools_for_request to select a ranked subset."
        )
    return items[:cap]
