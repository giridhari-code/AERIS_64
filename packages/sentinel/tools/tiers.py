"""Tool trust tiers — production policy for Backend tool runtime.

T0  Safe public facts (default on)
T1  Read-only local/remote (default on)
T2  Write / stateful (needs explicit enable)
T3  Code/shell execution (off unless NEUROFIELD_TOOLS_T3=1)
"""

from __future__ import annotations

import os
from typing import Dict, FrozenSet

# Canonical name → tier
TOOL_TIER: Dict[str, int] = {
    # T0
    "web_search": 0,
    "weather": 0,
    "time_now": 0,
    "time": 0,
    "calculator": 0,
    "calc": 0,
    "wikipedia": 0,
    "wiki": 0,
    "currency_convert": 0,
    "currency": 0,
    "tool_search": 0,
    # T1
    "web_fetch": 1,
    "file_search": 1,
    "git": 1,
    "rag_retrieve": 1,
    "rag": 1,
    "megha": 1,
    "image_generation": 1,
    # T2
    "ajeebwork_file": 2,
    "ajeebwork_plan": 2,
    "ajeebwork_subagents": 2,
    "ajeebwork_task": 2,
    "ajeebwork_memory": 2,
    "ajeebwork_document": 2,
    "ajeebwork_draft_edit": 2,
    "ajeebwork_skill": 2,
    "kalofact_create": 2,
    "kalofact_get": 2,
    "kalofact_list": 2,
    "kalofact_edit": 2,
    "kalofact_export": 2,
    "kalofact_share": 2,
    "dataset_builder": 2,
    "computer_use": 2,
    # T3
    "bash": 3,
    "code_execution": 3,
    "code_interpreter": 3,
}


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def max_allowed_tier() -> int:
    """Highest tier permitted in this process.

    NEUROFIELD_TOOLS_MAX_TIER=0|1|2|3 overrides.
    Else: T3 only if NEUROFIELD_TOOLS_T3=1; T2 if NEUROFIELD_TOOLS_T2=1 (default 1);
    T0–T1 always on unless max tier lowered.
    """
    raw = os.environ.get("NEUROFIELD_TOOLS_MAX_TIER", "").strip()
    if raw.isdigit():
        return max(0, min(3, int(raw)))
    if _env_truthy("NEUROFIELD_TOOLS_T3", "0"):
        return 3
    if _env_truthy("NEUROFIELD_TOOLS_T2", "1"):
        return 2
    return 1


def tier_of(name: str) -> int:
    return TOOL_TIER.get((name or "").strip().lower(), 1)


def is_allowed(name: str) -> bool:
    return tier_of(name) <= max_allowed_tier()


def deny_message(name: str) -> str:
    t = tier_of(name)
    max_t = max_allowed_tier()
    return (
        f"tool `{name}` is tier T{t}; process allows up to T{max_t}. "
        f"Set NEUROFIELD_TOOLS_MAX_TIER or NEUROFIELD_TOOLS_T3=1 for exec tools."
    )


def allowed_names(all_names: list[str] | FrozenSet[str] | set[str]) -> list[str]:
    return sorted(n for n in all_names if is_allowed(n))
