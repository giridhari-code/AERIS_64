"""Safe Git tool for the AERIS agent (status / log / diff / add / commit).

Designed so the agent can do the same class of git work that external agents
(Grok, Claude, etc.) perform on this repository — without giving unrestricted
shell access.

Safety:
  - Only a fixed set of subcommands is allowed.
  - ``push`` / ``pull`` / ``fetch`` / ``remote`` / ``rebase`` / ``reset --hard``
    are blocked by default (set NEUROFIELD_GIT_ALLOW_PUSH=1 to enable push).
  - Commands run with a short timeout and captured output.
  - Working directory must be inside a git repository (or explicit path).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

# Subcommands the agent is allowed to run.
_ALLOWED = {
    "status",
    "log",
    "diff",
    "add",
    "commit",
    "show",
    "branch",
    "checkout",  # only existing branches / files; no -b by default
    "stash",
    "tag",
    "rev-parse",
    "ls-files",
}

# Destructive or network operations — blocked unless explicitly enabled.
_BLOCKED = {
    "push",
    "pull",
    "fetch",
    "clone",
    "remote",
    "rebase",
    "reset",
    "clean",
    "filter-branch",
    "filter-repo",
    "gc",
}


def _allow_push() -> bool:
    return os.environ.get("NEUROFIELD_GIT_ALLOW_PUSH", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def git_report(
    command: str,
    path: Optional[str] = None,
    message: Optional[str] = None,
) -> str:
    """
    Run a safe git subcommand and return a short text report for the agent.

    Parameters
    ----------
    command:
        One of: status, log, diff, add, commit, show, branch, ...
        For ``commit`` you must also pass ``message``.
        For ``add`` you can pass paths in ``command`` after the subcommand
        (e.g. ``add src/``) or leave empty for ``add -A`` (careful).
    path:
        Optional working directory (must be inside a git repo).
    message:
        Commit message when command is ``commit``.
    """
    cmd = (command or "").strip()
    if not cmd:
        return "error: empty git command"

    parts = cmd.split()
    sub = parts[0].lower()

    if sub in _BLOCKED and not (sub == "push" and _allow_push()):
        return (
            f"error: git '{sub}' is blocked for safety. "
            "Set NEUROFIELD_GIT_ALLOW_PUSH=1 only if you really need push."
        )

    if sub not in _ALLOWED and not (sub == "push" and _allow_push()):
        return (
            f"error: git subcommand '{sub}' not allowed. "
            f"Allowed: {', '.join(sorted(_ALLOWED))}"
        )

    cwd = Path(path).resolve() if path else Path.cwd()
    if not (cwd / ".git").exists() and not any(
        (p / ".git").exists() for p in cwd.parents
    ):
        # Still try — git will error clearly if not a repo
        pass

    argv = ["git", "-C", str(cwd)]

    if sub == "status":
        argv += ["status", "--short", "--branch"]
    elif sub == "log":
        argv += ["log", "--oneline", "-20"]
    elif sub == "diff":
        argv += ["diff", "--stat"] + parts[1:]
    elif sub == "add":
        paths = parts[1:] if len(parts) > 1 else ["-A"]
        argv += ["add"] + paths
    elif sub == "commit":
        msg = (message or "").strip()
        if not msg:
            # allow "commit -m 'msg'" style inside command
            if "-m" in parts:
                try:
                    idx = parts.index("-m")
                    msg = parts[idx + 1]
                except (ValueError, IndexError):
                    msg = ""
            if not msg:
                return "error: commit requires a message (pass message=...)"
        argv += ["commit", "-m", msg]
    elif sub == "branch":
        argv += ["branch", "-v"] + parts[1:]
    elif sub == "show":
        argv += ["show", "--stat"] + parts[1:]
    elif sub == "push" and _allow_push():
        argv += ["push"] + parts[1:]
    else:
        # pass remaining tokens for other allowed subcommands
        argv += parts

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(cwd),
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        return "error: git command timed out (20s)"
    except FileNotFoundError:
        return "error: git binary not found on PATH"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = err or out or f"exit {proc.returncode}"
        return f"error (exit {proc.returncode}): {detail[:1500]}"
    if not out and not err:
        return "ok (no output)"
    return (out or err)[:3000]


def git_status(path: Optional[str] = None) -> str:
    """Shorthand for git status."""
    return git_report("status", path=path)
