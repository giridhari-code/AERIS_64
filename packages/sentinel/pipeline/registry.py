"""Stage registry — single source of truth for pipeline stages."""

from __future__ import annotations

from typing import Callable

# stage_name -> callable(ctx) -> None
STAGES: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        STAGES[name] = fn
        return fn

    return deco


def list_stages() -> list[str]:
    return list(STAGES.keys())
