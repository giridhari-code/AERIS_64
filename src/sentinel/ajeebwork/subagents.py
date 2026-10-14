"""Parallel sub-agent coordination with hard budgets.

Each sub-agent is a callable work item. Results are collected with a shared
timeout and concurrency limit from AjeebWorkConfig.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

from sentinel.ajeebwork.config import load_ajeebwork_config

WorkFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class SubAgentResult:
    name: str
    ok: bool
    detail: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_subagents(
    tasks: list[dict[str, Any]],
    worker: WorkFn,
    *,
    max_workers: Optional[int] = None,
    timeout_sec: Optional[int] = None,
) -> list[SubAgentResult]:
    """
    Run ``tasks`` in parallel. Each task dict must include ``name`` and any
    payload fields the ``worker`` understands.
    """
    cfg = load_ajeebwork_config()
    if not cfg.enabled:
        return [
            SubAgentResult(name="system", ok=False, detail="ajeebwork disabled", duration_ms=0.0)
        ]

    max_workers = min(
        int(max_workers if max_workers is not None else cfg.max_subagents),
        cfg.max_subagents,
        max(1, len(tasks)),
    )
    timeout_sec = int(timeout_sec if timeout_sec is not None else cfg.subagent_timeout_sec)
    tasks = tasks[: cfg.max_subagents]

    results: list[SubAgentResult] = []
    if not tasks:
        return results

    def _run_one(task: dict[str, Any]) -> SubAgentResult:
        name = str(task.get("name") or "subagent")
        t0 = time.time()
        try:
            out = worker(task)
            ok = bool(out.get("ok", True))
            detail = str(out.get("detail") or "")
        except Exception as e:
            ok = False
            detail = f"{type(e).__name__}: {e}"
        return SubAgentResult(
            name=name,
            ok=ok,
            detail=detail,
            duration_ms=(time.time() - t0) * 1000,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        try:
            for fut in as_completed(futures, timeout=timeout_sec):
                results.append(fut.result())
        except Exception as e:
            results.append(
                SubAgentResult(
                    name="coordinator",
                    ok=False,
                    detail=f"subagent pool error: {type(e).__name__}: {e}",
                    duration_ms=0.0,
                )
            )
    return results


def subagents_report(
    tasks: list[dict[str, Any]],
    worker: Optional[WorkFn] = None,
) -> str:
    if worker is None:
        lines = [f"SubAgents plan ({len(tasks)} tasks):"]
        for t in tasks:
            lines.append(f"  - {t.get('name')}: {t}")
        return "\n".join(lines)
    results = run_subagents(tasks, worker)
    lines = [f"SubAgents completed={sum(1 for r in results if r.ok)}/{len(results)}"]
    for r in results:
        lines.append(f"  [{('ok' if r.ok else 'err')}] {r.name} {r.duration_ms:.0f}ms {r.detail[:200]}")
    return "\n".join(lines)
