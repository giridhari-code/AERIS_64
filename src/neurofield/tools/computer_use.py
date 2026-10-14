"""Computer-use capability gate.

Full desktop control requires an attached backend (vendor computer-use API,
Playwright, or a locked-down VNC session). This module only exposes the
contract and remains disabled until NEUROFIELD_COMPUTER_USE=1 is set and a
backend is registered by the serving process.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from neurofield.tools.config import load_tools_config

_BACKEND: Optional[Callable[..., str]] = None

_SUPPORTED_ACTIONS = frozenset(
    {
        "screenshot",
        "click",
        "double_click",
        "type",
        "key",
        "scroll",
        "wait",
        "move",
        "drag",
    }
)


def register_computer_backend(fn: Callable[..., str]) -> None:
    """Serving layer calls this once to attach a real backend."""
    global _BACKEND
    _BACKEND = fn


def computer_use_available() -> bool:
    cfg = load_tools_config()
    return bool(cfg.computer_use_enabled and _BACKEND is not None)


def computer_use(
    action: str,
    coordinate: Optional[list[int]] = None,
    text: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Dispatch a computer-use action to the registered backend.

    Without a backend the call fails closed with a clear status string.
    """
    cfg = load_tools_config()
    act = (action or "").strip().lower()
    if not act:
        return "ComputerUse: empty action."
    if act not in _SUPPORTED_ACTIONS:
        return (
            f"ComputerUse: unsupported action '{action}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_ACTIONS))}."
        )

    if not cfg.computer_use_enabled:
        return (
            "ComputerUse: disabled. "
            "Set NEUROFIELD_COMPUTER_USE=1 and register a backend to enable."
        )

    if _BACKEND is None:
        return (
            "ComputerUse: enabled in config but no backend is registered. "
            "Call register_computer_backend() from the serving process."
        )

    try:
        return _BACKEND(action=act, coordinate=coordinate, text=text, **kwargs)
    except Exception as e:
        return f"ComputerUse backend error: {type(e).__name__}: {e}"
