"""Computer Use tool (high-level stub).

Full computer-use (screen, mouse, keyboard) requires a controlled desktop
environment (e.g. Anthropic computer-use, OpenAI computer-use, or a local
VNC/Playwright stack). This module exposes the interface and clearly reports
that the runtime is not attached yet.
"""

from __future__ import annotations

from typing import Any, Optional


def computer_use(
    action: str,
    coordinate: Optional[list[int]] = None,
    text: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Perform a computer-use action.

    Typical actions (when a backend is attached):
      - screenshot
      - click / double_click
      - type / key
      - scroll
      - wait

    Current build: no desktop backend attached.
    """
    action = (action or "").strip().lower()
    if not action:
        return "ComputerUse: empty action."

    supported = {
        "screenshot", "click", "double_click", "type", "key",
        "scroll", "wait", "move", "drag",
    }
    if action not in supported:
        return (
            f"ComputerUse: unknown action '{action}'. "
            f"Supported (when backend attached): {', '.join(sorted(supported))}"
        )

    detail = []
    if coordinate is not None:
        detail.append(f"coordinate={coordinate}")
    if text:
        detail.append(f"text={text[:80]}")
    for k, v in kwargs.items():
        detail.append(f"{k}={v}")

    extra = (", " + ", ".join(detail)) if detail else ""
    return (
        f"ComputerUse: action='{action}'{extra}\n"
        "No desktop / browser backend is attached in this build. "
        "Wire Playwright, a VNC session, or a vendor computer-use API to enable."
    )
