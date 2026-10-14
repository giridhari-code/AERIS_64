"""Configuration for AjeebWork / autonomous workspace features."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AjeebWorkConfig:
    enabled: bool = True
    # Workspace roots (comma-separated). Empty => repo root only.
    workspace_roots: tuple[str, ...] = ()
    max_read_bytes: int = 2_000_000
    max_write_bytes: int = 2_000_000
    max_batch_files: int = 50
    # Deletion requires explicit allow_delete on each call (never implicit).
    allow_delete_default: bool = False
    # Multi-step / sub-agent budgets
    max_plan_steps: int = 16
    max_subagents: int = 4
    subagent_timeout_sec: int = 120
    # Background tasks (in-process; cloud offline requires external runner)
    max_background_tasks: int = 32
    task_store_dir: str = "artifacts/ajeebwork_tasks"
    # Skills
    skills_dir: str = "artifacts/ajeebwork_skills"
    # Shared memory with chat sessions
    shared_memory_enabled: bool = True

    @classmethod
    def from_env(cls) -> "AjeebWorkConfig":
        roots_raw = (os.environ.get("NEUROFIELD_AJEEBWORK_ROOTS") or "").strip()
        roots = tuple(r.strip() for r in roots_raw.split(",") if r.strip())
        return cls(
            enabled=_env_bool("NEUROFIELD_AJEEBWORK", True),
            workspace_roots=roots,
            max_read_bytes=_env_int("NEUROFIELD_AJEEBWORK_MAX_READ", 2_000_000),
            max_write_bytes=_env_int("NEUROFIELD_AJEEBWORK_MAX_WRITE", 2_000_000),
            max_batch_files=_env_int("NEUROFIELD_AJEEBWORK_MAX_BATCH", 50),
            allow_delete_default=_env_bool("NEUROFIELD_AJEEBWORK_ALLOW_DELETE", False),
            max_plan_steps=_env_int("NEUROFIELD_AJEEBWORK_MAX_PLAN_STEPS", 16),
            max_subagents=_env_int("NEUROFIELD_AJEEBWORK_MAX_SUBAGENTS", 4),
            subagent_timeout_sec=_env_int("NEUROFIELD_AJEEBWORK_SUBAGENT_TIMEOUT", 120),
            max_background_tasks=_env_int("NEUROFIELD_AJEEBWORK_MAX_BG_TASKS", 32),
            task_store_dir=os.environ.get("NEUROFIELD_AJEEBWORK_TASK_DIR")
            or "artifacts/ajeebwork_tasks",
            skills_dir=os.environ.get("NEUROFIELD_AJEEBWORK_SKILLS_DIR")
            or "artifacts/ajeebwork_skills",
            shared_memory_enabled=_env_bool("NEUROFIELD_AJEEBWORK_SHARED_MEMORY", True),
        )


_CFG: Optional[AjeebWorkConfig] = None


def load_ajeebwork_config(force: bool = False) -> AjeebWorkConfig:
    global _CFG
    if _CFG is None or force:
        _CFG = AjeebWorkConfig.from_env()
    return _CFG


def resolve_workspace_roots(cfg: Optional[AjeebWorkConfig] = None) -> list[Path]:
    cfg = cfg or load_ajeebwork_config()
    if cfg.workspace_roots:
        return [Path(r).expanduser().resolve() for r in cfg.workspace_roots]
    # Default: repository root (pyproject / .git walk)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return [parent]
    return [Path.cwd().resolve()]
