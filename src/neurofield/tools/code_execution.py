"""Isolated Python code execution (Code Interpreter).

Uses the production SandboxExecutor. Network is disabled by default.
Session injection is preferred in multi-tenant serve; a process-local
session is used only when no session is provided.
"""

from __future__ import annotations

import threading
from typing import Optional

from neurofield.sandbox.executor import SandboxConfig
from neurofield.sandbox.session import SessionSandbox
from neurofield.tools.config import load_tools_config

_lock = threading.Lock()
_fallback_session: Optional[SessionSandbox] = None


def _fallback() -> SessionSandbox:
    global _fallback_session
    with _lock:
        if _fallback_session is None:
            cfg = load_tools_config()
            _fallback_session = SessionSandbox(
                "tool-code-execution",
                cfg=SandboxConfig(
                    timeout_sec=cfg.code_timeout_sec,
                    max_output_bytes=cfg.code_max_output_bytes,
                    max_memory_mb=cfg.code_max_memory_mb,
                    allow_network=False,
                ),
                max_calls=256,
            )
        return _fallback_session


def code_execution(
    code: str,
    language: str = "python",
    session: Optional[SessionSandbox] = None,
) -> str:
    """
    Execute Python source in the sandbox and return a bounded report string.

    Parameters
    ----------
    code : str
        Source to run.
    language : str
        Only ``python`` is supported in this build.
    session : SessionSandbox, optional
        Per-user / per-chat sandbox. If omitted, a process-local fallback is used.
    """
    cfg = load_tools_config()
    source = (code or "").strip()
    if not source:
        return "CodeExecution: empty code."
    if len(source) > cfg.code_max_source_chars:
        return (
            f"CodeExecution: source exceeds limit "
            f"({len(source)} > {cfg.code_max_source_chars} chars)."
        )

    lang = (language or "python").lower().strip()
    if lang not in {"python", "python3", "py"}:
        return f"CodeExecution: unsupported language '{language}' (supported: python)."

    runner = session or _fallback()
    result = runner.run_python(source)

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
    return f"CodeExecution ({status}, {result.duration_ms:.0f}ms):\n{body}"


code_interpreter = code_execution
