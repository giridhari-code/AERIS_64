"""
Isolated subprocess sandbox for NeuroField tool-use (production).

Limits (aligned with paper Section 6 controls):
  - wall-clock timeout, RLIMIT_CPU / AS / NPROC / FSIZE / NOFILE
  - binary allow-list, checked on a shell-free argv (no ``;`` ``&&`` ``|`` tricks)
  - no network by default: a fresh network namespace (``unshare``) when the host
    allows it, plus dead proxy env vars. Set ``require_isolation=True`` to refuse
    to run when the namespace cannot be created (fail closed).
  - per-session work_dir (cwd + HOME + TMPDIR); removed when the session is evicted
  - every call recorded in an audit list

IMPORTANT: this is defence in depth, NOT a security boundary. ``python3`` is allowed,
so run the server itself in a container / gVisor / nsjail with no secrets and no
network if untrusted users can reach the tool endpoints.
"""

from __future__ import annotations

import hashlib
import os
import resource
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SandboxConfig:
    timeout_sec: float = 8.0
    max_output_bytes: int = 32_768
    max_memory_mb: int = 256
    max_cpu_sec: int = 5
    max_procs: int = 128          # RLIMIT_NPROC (0 = off); not enforced for uid 0
    max_file_mb: int = 16         # RLIMIT_FSIZE
    max_open_files: int = 64      # RLIMIT_NOFILE
    allow_network: bool = False
    require_isolation: bool = False  # refuse to run offline commands without a netns
    allowed_bins: tuple[str, ...] = (
        "python3",
        "python",
        "cat",
        "head",
        "tail",
        "wc",
        "ls",
        "echo",
        "printf",
        "sort",
        "uniq",
        "grep",
        "sed",
        "awk",
        "cut",
        "tr",
        "date",
        "pwd",
        "whoami",
        "uname",
        "true",
        "false",
    )
    work_root: Optional[str] = None


@dataclass
class SandboxResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool
    command: str
    work_dir: str
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "command": self.command,
            "work_dir": self.work_dir,
            "audit": self.audit,
        }


def _set_limit(which: int, value: int) -> None:
    try:
        resource.setrlimit(which, (value, value))
    except (ValueError, resource.error, OSError):
        pass


def _preexec_limits(cfg: SandboxConfig) -> None:
    """Runs in the child before exec. Keep it tiny and async-signal-safe-ish."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cfg.max_cpu_sec, cfg.max_cpu_sec + 1))
    except (ValueError, resource.error, OSError):
        pass
    _set_limit(resource.RLIMIT_AS, cfg.max_memory_mb * 1024 * 1024)
    _set_limit(resource.RLIMIT_CORE, 0)
    if cfg.max_file_mb > 0:
        _set_limit(resource.RLIMIT_FSIZE, cfg.max_file_mb * 1024 * 1024)
    if cfg.max_open_files > 0:
        _set_limit(resource.RLIMIT_NOFILE, cfg.max_open_files)
    if cfg.max_procs > 0:
        _set_limit(resource.RLIMIT_NPROC, cfg.max_procs)
    # new session / process group is requested via Popen(start_new_session=True)


_NET_ISOLATION_PREFIX: Optional[list[str]] = None


def _net_isolation_prefix() -> list[str]:
    """argv prefix that runs a command in an empty network namespace, or [] if unavailable."""
    global _NET_ISOLATION_PREFIX
    if _NET_ISOLATION_PREFIX is not None:
        return _NET_ISOLATION_PREFIX
    prefix: list[str] = []
    exe = shutil.which("unshare")
    if exe:
        for flags in (["--net"], ["--user", "--map-root-user", "--net"]):
            try:
                r = subprocess.run([exe, *flags, "--", "true"], capture_output=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                prefix = [exe, *flags, "--"]
                break
    _NET_ISOLATION_PREFIX = prefix
    return prefix


class SandboxExecutor:
    def __init__(self, cfg: Optional[SandboxConfig] = None, session_id: str = "default"):
        self.cfg = cfg or SandboxConfig()
        self.session_id = session_id
        root = Path(self.cfg.work_root or tempfile.gettempdir()) / "neurofield_sandbox"
        root.mkdir(parents=True, exist_ok=True)
        self.work_dir = root / self._safe_id(session_id)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._call_count = 0
        self.audit_log: list[dict[str, Any]] = []

    @staticmethod
    def _safe_id(sid: str) -> str:
        """Filesystem-safe, collision-free dir name (digest of the FULL id)."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)[:48] or "default"
        digest = hashlib.sha256(sid.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"{safe}-{digest}"

    def _truncate(self, data: bytes) -> str:
        if len(data) > self.cfg.max_output_bytes:
            data = data[: self.cfg.max_output_bytes] + b"\n...[truncated]..."
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _parse(command: str) -> Optional[list[str]]:
        try:
            argv = shlex.split(command)
        except ValueError:
            return None
        return argv or None

    def _bin_allowed(self, cmd: str) -> bool:
        argv = self._parse(cmd)
        # bare names only: "./x" or "/bin/cat" are refused so cwd/absolute paths can't dodge the list
        return bool(argv) and "/" not in argv[0] and argv[0] in self.cfg.allowed_bins

    def run_shell(self, command: str, env_extra: Optional[dict[str, str]] = None) -> SandboxResult:
        self._call_count += 1
        if not command or not command.strip():
            result = SandboxResult(
                ok=False, exit_code=-1, stdout="", stderr="empty command",
                duration_ms=0.0, timed_out=False, command=command or "",
                work_dir=str(self.work_dir), audit={"reason": "empty", "call": self._call_count},
            )
            self.audit_log.append(result.to_dict())
            return result
        if not self._bin_allowed(command):
            result = SandboxResult(
                ok=False, exit_code=-1, stdout="",
                stderr=(
                    "unparseable command (unbalanced quotes)"
                    if self._parse(command) is None
                    else f"binary not in allow-list: {self._parse(command)[0]}"
                ),
                duration_ms=0.0, timed_out=False, command=command,
                work_dir=str(self.work_dir), audit={"reason": "deny_bin", "call": self._call_count},
            )
            self.audit_log.append(result.to_dict())
            return result

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.work_dir),
            "TMPDIR": str(self.work_dir),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if not self.cfg.allow_network:
            env["no_proxy"] = "*"
            env["NO_PROXY"] = "*"
            env["http_proxy"] = "http://127.0.0.1:0"
            env["https_proxy"] = "http://127.0.0.1:0"
        if env_extra:
            env.update(env_extra)

        argv = self._parse(command) or []
        resolved = shutil.which(argv[0], path=env["PATH"])
        if resolved is None:
            result = SandboxResult(
                ok=False, exit_code=-1, stdout="", stderr=f"binary not found: {argv[0]}",
                duration_ms=0.0, timed_out=False, command=command,
                work_dir=str(self.work_dir), audit={"reason": "not_found", "call": self._call_count},
            )
            self.audit_log.append(result.to_dict())
            return result
        argv[0] = resolved

        net_isolated = False
        if not self.cfg.allow_network:
            prefix = _net_isolation_prefix()
            if prefix:
                argv = prefix + argv
                net_isolated = True
            elif self.cfg.require_isolation:
                result = SandboxResult(
                    ok=False, exit_code=-1, stdout="",
                    stderr="network isolation unavailable on this host (require_isolation=True)",
                    duration_ms=0.0, timed_out=False, command=command,
                    work_dir=str(self.work_dir),
                    audit={"reason": "no_isolation", "call": self._call_count},
                )
                self.audit_log.append(result.to_dict())
                return result

        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.Popen(
                argv,  # no shell: ';', '&&', '|', '$(...)' are literal arguments
                cwd=str(self.work_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=lambda: _preexec_limits(self.cfg),
            )
            try:
                out_b, err_b = proc.communicate(timeout=self.cfg.timeout_sec)
                code = proc.returncode if proc.returncode is not None else -1
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                out_b, err_b = proc.communicate(timeout=1)
                code = -9
        except Exception as e:
            duration = (time.monotonic() - t0) * 1000
            result = SandboxResult(
                ok=False, exit_code=-1, stdout="", stderr=str(e),
                duration_ms=duration, timed_out=False, command=command,
                work_dir=str(self.work_dir),
                audit={"reason": "spawn_error", "call": self._call_count},
            )
            self.audit_log.append(result.to_dict())
            return result

        duration = (time.monotonic() - t0) * 1000
        result = SandboxResult(
            ok=(code == 0 and not timed_out),
            exit_code=code,
            stdout=self._truncate(out_b or b""),
            stderr=self._truncate(err_b or b""),
            duration_ms=duration,
            timed_out=timed_out,
            command=command,
            work_dir=str(self.work_dir),
            audit={
                "call": self._call_count,
                "session_id": self.session_id,
                "allow_network": self.cfg.allow_network,
                "net_isolated": net_isolated,
            },
        )
        self.audit_log.append(result.to_dict())
        return result

    def run_python(self, code: str) -> SandboxResult:
        script = self.work_dir / f"_nf_snippet_{self._call_count + 1}.py"
        script.write_text(code, encoding="utf-8")
        try:
            return self.run_shell(f"python3 -I -B {script.name}")
        finally:
            try:
                script.unlink(missing_ok=True)
            except OSError:
                pass

    def write_file(self, relative_path: str, content: str) -> Path:
        base = self.work_dir.resolve()
        path = (base / relative_path).resolve()
        if not path.is_relative_to(base):  # startswith() let "<dir>_evil" through
            raise ValueError("path escapes sandbox work_dir")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def reset_work_dir(self) -> None:
        import shutil
        for p in self.work_dir.iterdir():
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
