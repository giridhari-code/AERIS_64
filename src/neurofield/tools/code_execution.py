"""Code Interpreter / Code Execution tool (uses NeuroField sandbox)."""

from __future__ import annotations

from typing import Optional

from neurofield.sandbox.session import SessionSandbox
from neurofield.sandbox.executor import SandboxConfig


# Lightweight shared session for one-off tool calls (serving can inject real session)
_default_session: Optional[SessionSandbox] = None


def _get_session() -> SessionSandbox:
    global _default_session
    if _default_session is None:
        cfg = SandboxConfig(
            timeout_sec=12.0,
            max_output_bytes=48_000,
            max_memory_mb=384,
            allow_network=False,
        )
        _default_session = SessionSandbox("tool-code-exec", cfg=cfg, max_calls=64)
    return _default_session


def code_execution(code: str, language: str = "python") -> str:
    """
    Run short code in an isolated sandbox and return stdout/stderr.
    Currently supports Python (the primary interpreter for this project).
    """
    code = (code or "").strip()
    if not code:
        return "CodeExecution: empty code."
    if len(code) > 12_000:
        return "CodeExecution: code too long (max ~12k chars)."

    lang = (language or "python").lower().strip()
    if lang not in {"python", "python3", "py"}:
        return f"CodeExecution: language '{language}' not supported yet (only python)."

    session = _get_session()
    result = session.run_python(code)

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
    return f"CodeExecution ({status}, {result.duration_ms:.0f}ms):\n{body}"


# Alias used by some catalogs
code_interpreter = code_execution
