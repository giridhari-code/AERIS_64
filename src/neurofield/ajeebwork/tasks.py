"""Background and scheduled task registry (in-process).

Long-running work can be tracked with progress callbacks. True offline/cloud
execution when the device is off requires an external worker (queue, cron, or
hosted runner); this module persists task records so such a worker can resume.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Optional

from neurofield.ajeebwork.config import load_ajeebwork_config

_lock = threading.Lock()
_tasks: dict[str, "TaskRecord"] = {}


@dataclass
class TaskRecord:
    task_id: str
    name: str
    status: str  # queued | running | completed | failed | cancelled
    created_at: float
    updated_at: float
    progress: float = 0.0
    detail: str = ""
    schedule: Optional[str] = None  # cron-like label or ISO interval note
    result: Optional[str] = None
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _store_dir() -> Path:
    cfg = load_ajeebwork_config()
    d = Path(cfg.task_store_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist(rec: TaskRecord) -> None:
    path = _store_dir() / f"{rec.task_id}.json"
    path.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")


def create_task(
    name: str,
    *,
    schedule: Optional[str] = None,
    detail: str = "",
) -> TaskRecord:
    cfg = load_ajeebwork_config()
    with _lock:
        if len(_tasks) >= cfg.max_background_tasks:
            raise RuntimeError(
                f"background task limit reached ({cfg.max_background_tasks})"
            )
        tid = uuid.uuid4().hex[:12]
        now = time.time()
        rec = TaskRecord(
            task_id=tid,
            name=name,
            status="queued",
            created_at=now,
            updated_at=now,
            detail=detail,
            schedule=schedule,
        )
        _tasks[tid] = rec
        _persist(rec)
        return rec


def update_task(
    task_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    detail: Optional[str] = None,
    result: Optional[str] = None,
    reasoning_line: Optional[str] = None,
) -> Optional[TaskRecord]:
    with _lock:
        rec = _tasks.get(task_id)
        if rec is None:
            # try disk
            path = _store_dir() / f"{task_id}.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                rec = TaskRecord(**{k: data[k] for k in TaskRecord.__dataclass_fields__})
                _tasks[task_id] = rec
            else:
                return None
        if status is not None:
            rec.status = status
        if progress is not None:
            rec.progress = max(0.0, min(1.0, float(progress)))
        if detail is not None:
            rec.detail = detail
        if result is not None:
            rec.result = result
        if reasoning_line:
            rec.reasoning.append(reasoning_line)
        rec.updated_at = time.time()
        _persist(rec)
        return rec


def get_task(task_id: str) -> Optional[TaskRecord]:
    with _lock:
        rec = _tasks.get(task_id)
        if rec:
            return rec
    path = _store_dir() / f"{task_id}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return TaskRecord(**{k: data.get(k, getattr(TaskRecord, k, None)) for k in TaskRecord.__dataclass_fields__})
    return None


def list_tasks(limit: int = 20) -> list[TaskRecord]:
    with _lock:
        items = sorted(_tasks.values(), key=lambda r: r.updated_at, reverse=True)
        if items:
            return items[:limit]
    # load from disk
    out: list[TaskRecord] = []
    for path in sorted(_store_dir().glob("*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append(TaskRecord(**{k: data.get(k) for k in TaskRecord.__dataclass_fields__ if k in data}))
        except Exception:
            continue
    return out


def run_task_inline(
    name: str,
    work: Callable[[TaskRecord], str],
    *,
    schedule: Optional[str] = None,
) -> TaskRecord:
    """Run work synchronously while updating progress/reasoning for transparency."""
    rec = create_task(name, schedule=schedule)
    update_task(rec.task_id, status="running", progress=0.05, reasoning_line="started")
    try:
        result = work(rec)
        update_task(
            rec.task_id,
            status="completed",
            progress=1.0,
            result=result,
            reasoning_line="completed",
        )
    except Exception as e:
        update_task(
            rec.task_id,
            status="failed",
            detail=f"{type(e).__name__}: {e}",
            reasoning_line="failed",
        )
    return get_task(rec.task_id) or rec


def tasks_report(task_id: Optional[str] = None) -> str:
    if task_id:
        rec = get_task(task_id)
        if not rec:
            return f"Tasks: unknown task_id={task_id}"
        return (
            f"Task {rec.task_id} name={rec.name} status={rec.status} "
            f"progress={rec.progress:.0%} detail={rec.detail}\n"
            f"reasoning: {rec.reasoning}\nresult: {rec.result or ''}"
        )
    items = list_tasks()
    if not items:
        return "Tasks: no background tasks."
    lines = ["Tasks:"]
    for r in items:
        lines.append(
            f"  {r.task_id} {r.status} {r.progress:.0%} {r.name} {r.schedule or ''}"
        )
    return "\n".join(lines)
