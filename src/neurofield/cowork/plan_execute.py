"""Autonomous multi-step plan execution with structured recovery.

The planner produces an ordered list of steps. The executor runs each step
through a provided ``run_step`` callback, records outcomes, and can retry
failed steps within budget. This layer does not invent tool calls; the
serving/agent layer supplies ``run_step``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

from neurofield.cowork.config import load_cowork_config

RunStepFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class PlanStep:
    id: str
    title: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | ok | error | skipped
    detail: str = ""
    attempts: int = 0


@dataclass
class PlanResult:
    plan_id: str
    objective: str
    steps: list[PlanStep]
    stopped_reason: str
    ok: bool
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "steps": [asdict(s) for s in self.steps],
            "stopped_reason": self.stopped_reason,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 2),
        }


def build_plan(objective: str, step_specs: list[dict[str, Any]]) -> list[PlanStep]:
    cfg = load_cowork_config()
    if len(step_specs) > cfg.max_plan_steps:
        step_specs = step_specs[: cfg.max_plan_steps]
    steps: list[PlanStep] = []
    for i, spec in enumerate(step_specs):
        steps.append(
            PlanStep(
                id=str(spec.get("id") or f"s{i+1}"),
                title=str(spec.get("title") or spec.get("action") or f"step-{i+1}"),
                action=str(spec.get("action") or "noop"),
                args=dict(spec.get("args") or {}),
            )
        )
    return steps


def execute_plan(
    objective: str,
    step_specs: list[dict[str, Any]],
    run_step: RunStepFn,
    *,
    max_retries: int = 1,
) -> PlanResult:
    """
    Execute steps sequentially. ``run_step`` receives
    ``{"action", "args", "step_id", "title"}`` and must return
    ``{"ok": bool, "detail": str}``.
    """
    t0 = time.time()
    plan_id = uuid.uuid4().hex[:12]
    steps = build_plan(objective, step_specs)
    if not steps:
        return PlanResult(
            plan_id=plan_id,
            objective=objective,
            steps=[],
            stopped_reason="empty_plan",
            ok=False,
            duration_ms=(time.time() - t0) * 1000,
        )

    for step in steps:
        success = False
        last_detail = ""
        while step.attempts <= max_retries and not success:
            step.attempts += 1
            step.status = "running"
            try:
                out = run_step(
                    {
                        "action": step.action,
                        "args": step.args,
                        "step_id": step.id,
                        "title": step.title,
                    }
                )
                success = bool(out.get("ok"))
                last_detail = str(out.get("detail") or "")
            except Exception as e:
                success = False
                last_detail = f"{type(e).__name__}: {e}"
            if success:
                step.status = "ok"
                step.detail = last_detail
            else:
                step.status = "error"
                step.detail = last_detail
        if not success:
            # Stop on hard failure after retries
            return PlanResult(
                plan_id=plan_id,
                objective=objective,
                steps=steps,
                stopped_reason=f"step_failed:{step.id}",
                ok=False,
                duration_ms=(time.time() - t0) * 1000,
            )

    return PlanResult(
        plan_id=plan_id,
        objective=objective,
        steps=steps,
        stopped_reason="completed",
        ok=True,
        duration_ms=(time.time() - t0) * 1000,
    )


def plan_execute_report(
    objective: str,
    step_specs: list[dict[str, Any]],
    run_step: Optional[RunStepFn] = None,
) -> str:
    """String report for tool/agent injection."""
    if run_step is None:
        # Dry-run: only structure the plan
        steps = build_plan(objective, step_specs)
        lines = [f"Plan ({len(steps)} steps) for: {objective}"]
        for s in steps:
            lines.append(f"  - [{s.id}] {s.title} action={s.action}")
        return "\n".join(lines)
    result = execute_plan(objective, step_specs, run_step)
    lines = [
        f"PlanExecute id={result.plan_id} ok={result.ok} "
        f"reason={result.stopped_reason} ({result.duration_ms:.0f}ms)",
        f"Objective: {objective}",
    ]
    for s in result.steps:
        lines.append(
            f"  [{s.status}] {s.id} {s.title} attempts={s.attempts} {s.detail[:200]}"
        )
    return "\n".join(lines)
