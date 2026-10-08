"""Per-session sandbox registry bound to NeuroField inference sessions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .executor import SandboxConfig, SandboxExecutor, SandboxResult


@dataclass
class SessionSandbox:
    session_id: str
    cfg: SandboxConfig = field(default_factory=SandboxConfig)
    max_calls: int = 32
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.executor = SandboxExecutor(self.cfg, session_id=self.session_id)
        self._lock = threading.Lock()
        self.calls = 0

    def run_shell(self, command: str) -> SandboxResult:
        with self._lock:
            self.last_used = time.time()
            if self.calls >= self.max_calls:
                return SandboxResult(
                    ok=False, exit_code=-1, stdout="",
                    stderr=f"session tool-call budget exhausted ({self.max_calls})",
                    duration_ms=0.0, timed_out=False, command=command,
                    work_dir=str(self.executor.work_dir),
                    audit={"reason": "budget"},
                )
            self.calls += 1
            return self.executor.run_shell(command)

    def run_python(self, code: str) -> SandboxResult:
        with self._lock:
            self.last_used = time.time()
            if self.calls >= self.max_calls:
                return SandboxResult(
                    ok=False, exit_code=-1, stdout="",
                    stderr=f"session tool-call budget exhausted ({self.max_calls})",
                    duration_ms=0.0, timed_out=False, command="<python>",
                    work_dir=str(self.executor.work_dir),
                    audit={"reason": "budget"},
                )
            self.calls += 1
            return self.executor.run_python(code)

    def reset(self) -> None:
        with self._lock:
            self.executor.reset_work_dir()
            self.executor.audit_log.clear()
            self.calls = 0
            self.last_used = time.time()

    def audit(self) -> list[dict[str, Any]]:
        return list(self.executor.audit_log)


class SandboxRegistry:
    def __init__(
        self,
        cfg: Optional[SandboxConfig] = None,
        max_sessions: int = 64,
        ttl_sec: float = 1800.0,
        max_calls_per_session: int = 32,
    ):
        self.cfg = cfg or SandboxConfig()
        self.max_sessions = max_sessions
        self.ttl_sec = ttl_sec
        self.max_calls = max_calls_per_session
        self._sessions: dict[str, SessionSandbox] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionSandbox:
        with self._lock:
            self._evict_unlocked()
            sb = self._sessions.get(session_id)
            if sb is None:
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                    del self._sessions[oldest.session_id]
                sb = SessionSandbox(
                    session_id=session_id,
                    cfg=self.cfg,
                    max_calls=self.max_calls,
                )
                self._sessions[session_id] = sb
            sb.last_used = time.time()
            return sb

    def reset(self, session_id: str) -> None:
        with self._lock:
            sb = self._sessions.get(session_id)
            if sb is not None:
                sb.reset()

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_unlocked(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items() if now - s.last_used > self.ttl_sec]
        for sid in dead:
            del self._sessions[sid]
