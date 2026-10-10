"""
Free END-TO-END tasks — open prompts, full pipeline, optional weak checks.

"Free" = not multiple-choice; natural language in → model/server out.
Tiny checkpoints will fail many free tasks; the harness still measures the path.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class TaskResult:
    id: str
    category: str
    prompt: str
    output: str
    passed: Optional[bool]
    latency_ms: float
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    tasks: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        tasks.append(json.loads(ln))
    return tasks


def _check(output: str, task: dict[str, Any]) -> Optional[bool]:
    needles = task.get("expect_contains_any") or []
    if not needles:
        return None  # free task, no auto grade
    out = (output or "").lower()
    return any(str(n).lower() in out for n in needles)


def run_task(
    task: dict[str, Any],
    generate_fn: Callable[..., str],
    **gen_kwargs: Any,
) -> TaskResult:
    """
    generate_fn(prompt, max_tokens=..., **kwargs) -> str
    """
    prompt = str(task.get("prompt") or "")
    max_tokens = int(task.get("max_tokens") or 64)
    t0 = time.perf_counter()
    try:
        output = generate_fn(prompt, max_tokens=max_tokens, **gen_kwargs)
    except TypeError:
        output = generate_fn(prompt)
    except Exception as e:
        output = f"[error] {e}"
    dt = (time.perf_counter() - t0) * 1000.0
    passed = _check(output, task)
    return TaskResult(
        id=str(task.get("id") or "task"),
        category=str(task.get("category") or "free"),
        prompt=prompt,
        output=(output or "")[:2000],
        passed=passed,
        latency_ms=round(dt, 2),
        notes=str(task.get("notes") or ""),
        meta={"max_tokens": max_tokens},
    )


def run_task_file(
    path: str | Path,
    generate_fn: Callable[..., str],
    skip_placeholder: bool = True,
    **gen_kwargs: Any,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for task in load_tasks(path):
        if skip_placeholder and "PLACEHOLDER" in str(task.get("prompt") or ""):
            continue
        results.append(run_task(task, generate_fn, **gen_kwargs))
    return results
