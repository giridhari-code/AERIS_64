"""Restricted Bash tool (uses NeuroField sandbox allow-list)."""

from __future__ import annotations

from typing import Optional

from neurofield.sandbox.session import SessionSandbox
from neurofield.sandbox.executor import SandboxConfig


_default_session: Optional[SessionSandbox] = None


def _get_session() -> SessionSandbox:
    global _default_session
    if _default_session is None:
        cfg = SandboxConfig(
            timeout_sec=10.0,
            max_output_bytes=40_000,
            max_memory_mb=256,
            allow_network=False,
        )
        _default_session = SessionSandbox("tool-bash", cfg=cfg, max_calls=64)
    return _default_session


def bash(command: str) -> str:
    """
    Run a single shell command inside the sandbox.
    Only allow-listed binaries are permitted (see SandboxConfig.allowed_bins).
    """
    command = (command or "").strip()
    if not command:
        return "Bash: empty command."
    if len(command) > 2_000:
        return "Bash: command too long."

    # Soft block of obviously dangerous patterns (sandbox also enforces)
    lowered = command.lower()
    blocked = ("rm -rf /", "mkfs", ":(){", "dd if=", ">/dev/sd", "chmod 777 /")
    if any(b in lowered for b in blocked):
        return "Bash: command blocked by safety filter."

    session = _get_session()
    result = session.run_shell(command)

    parts = []
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
