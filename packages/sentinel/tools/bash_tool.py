"""Restricted shell execution via the production sandbox.

Commands are executed through SandboxExecutor (argv allow-list, resource
limits, optional network namespace). Shell metacharacters are not expanded
as a login shell would; pass a single allow-listed binary + args.
"""

from __future__ import annotations

import threading
from typing import Optional

from sentinel.sandbox.executor import SandboxConfig
from sentinel.sandbox.session import SessionSandbox
from sentinel.tools.config import load_tools_config

_lock = threading.Lock()
_fallback_session: Optional[SessionSandbox] = None

# Defence-in-depth patterns; primary enforcement is SandboxConfig.allowed_bins
_BLOCKED_SUBSTRINGS = (
    "rm -rf /",
    "mkfs",
    ":(){",
    "dd if=",
    ">/dev/sd",
    "chmod -R 777 /",
    "curl ",
    "wget ",
)


def _fallback() -> SessionSandbox:
    global _fallback_session
    with _lock:
        if _fallback_session is None:
            cfg = load_tools_config()
            _fallback_session = SessionSandbox(
                "tool-bash",
                cfg=SandboxConfig(
                    timeout_sec=cfg.bash_timeout_sec,
                    max_output_bytes=cfg.bash_max_output_bytes,
                    max_memory_mb=cfg.bash_max_memory_mb,
                    allow_network=False,
                ),
                max_calls=256,
            )
        return _fallback_session


def bash(
    command: str,
    session: Optional[SessionSandbox] = None,
) -> str:
    """
    Run a single command string inside the sandbox.

    Parameters
    ----------
    command : str
        Command text. Prefer simple forms such as ``ls -la`` or ``pwd``.
    session : SessionSandbox, optional
        Per-tenant sandbox when serving multi-user traffic.
    """
    cfg = load_tools_config()
    cmd = (command or "").strip()
    if not cmd:
        return "Bash: empty command."
    if len(cmd) > cfg.bash_max_command_chars:
        return (
            f"Bash: command exceeds limit "
            f"({len(cmd)} > {cfg.bash_max_command_chars} chars)."
        )

    lowered = cmd.lower()
    for blocked in _BLOCKED_SUBSTRINGS:
        if blocked in lowered:
            return "Bash: command rejected by safety filter."

    runner = session or _fallback()
    result = runner.run_shell(cmd)

    parts: list[str] = []
    if result.stdout.strip():
        parts.append(result.stdout.strip())
    if result.stderr.strip():
        parts.append(f"[stderr]\n{result.stderr.strip()}")
    if result.timed_out:
        parts.append("[timed out]")
    if not result.ok and not parts:
        parts.append(f"exit_code={result.exit_code}")

    body = "\n".join(parts) if parts else "(no output)"
    status = "ok" if result.ok else "error"
    return f"Bash ({status}, {result.duration_ms:.0f}ms):\n{body}"
