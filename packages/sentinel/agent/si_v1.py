"""
SI_v1 — multi-step agent: plan → tool → verify → answer.

Builds on SI_v0 (single enrich). Here the agent:
  1. Classifies a simple plan from the user text (deterministic rules)
  2. Runs each step with allow-listed tools
  3. Verifies outputs (non-empty, no hard error prefix)
  4. Optionally polishes with generate_fn
  5. Stops on max_steps / fail budget / success

No LM tool-JSON required (reliable). LM only optional for final wording.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

from sentinel.tools.calculator import calculate
from sentinel.tools.timeutil import now_report
from sentinel.tools.weather import weather_report
from sentinel.tools.web_search import web_search_report
from sentinel.tools.wikipedia import wikipedia_summary
from sentinel.tools.currency import convert_currency
from sentinel.modules.bhutu import bhutu_decide

GenerateFn = Callable[[str], str]


@dataclass
class StepResult:
    id: str
    action: str
    ok: bool
    detail: str
    ms: float


@dataclass
class SI_v1Config:
    max_steps: int = 6
    max_failures: int = 2
    tool_timeout_s: float = 15.0


@dataclass
class SI_v1Result:
    answer: str
    plan: list[dict[str, str]]
    steps: list[StepResult] = field(default_factory=list)
    stopped_reason: str = ""
    latency_ms: float = 0.0
    facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- planners (deterministic) ---------------------------------------------

_CALC_EXPR = re.compile(
    r"(?:calculate|compute|what\s+is|kitna\s+hota)\s+([0-9\.\s\+\-\*\/\(\)%]+)",
    re.I,
)
_CALC_BARE = re.compile(r"^\s*([0-9]+(?:\s*[\+\-\*\/]\s*[0-9]+)+)\s*$")
_WEATHER = re.compile(r"\b(weather|temperature|mausam|forecast)\b", re.I)
_TIME = re.compile(r"\b(what\s+time|current\s+time|time\s+now|aaj\s+(ki\s+)?tarikh|date\s+today)\b", re.I)
_SEARCH = re.compile(
    r"\b(search|google|look\s*up|latest\s+news|current\s+news|headline|news|khabar|"
    r"trending|twitter|what.?s\s+happening|kya\s+chal\s+raha)\b",
    re.I,
)
_WIKI = re.compile(r"\b(who\s+is|what\s+is|wikipedia|explain)\b", re.I)
_FX = re.compile(
    r"\b(?:convert|change)\s+([0-9.]+)\s*([A-Za-z]{3})\s+(?:to|in)\s+([A-Za-z]{3})\b",
    re.I,
)


def build_plan(user_text: str) -> list[dict[str, str]]:
    """Return ordered steps: [{id, action, arg}, ...]."""
    p = user_text.strip()
    steps: list[dict[str, str]] = []

    m = _CALC_EXPR.search(p) or _CALC_BARE.search(p)
    if m:
        expr = m.group(1).strip()
        steps.append({"id": "s1", "action": "calculator", "arg": expr})
        steps.append({"id": "s2", "action": "verify_nonempty", "arg": "s1"})
        steps.append({"id": "s3", "action": "compose_answer", "arg": ""})
        return steps

    mfx = _FX.search(p)
    if mfx:
        arg = f"{mfx.group(1)} {mfx.group(2)} {mfx.group(3)}"
        steps.append({"id": "s1", "action": "currency", "arg": arg})
        steps.append({"id": "s2", "action": "verify_nonempty", "arg": "s1"})
        steps.append({"id": "s3", "action": "compose_answer", "arg": ""})
        return steps

    if _TIME.search(p):
        steps.append({"id": "s1", "action": "time", "arg": ""})
        steps.append({"id": "s2", "action": "verify_nonempty", "arg": "s1"})
        steps.append({"id": "s3", "action": "compose_answer", "arg": ""})
        return steps

    if _WEATHER.search(p):
        # naive location: last capitalized word or "local"
        loc = "local"
        ml = re.search(r"\bin\s+([A-Za-z][A-Za-z\s]{1,40})", p)
        if ml:
            loc = ml.group(1).strip()
        steps.append({"id": "s1", "action": "weather", "arg": loc})
        steps.append({"id": "s2", "action": "verify_nonempty", "arg": "s1"})
        steps.append({"id": "s3", "action": "compose_answer", "arg": ""})
        return steps

    if _SEARCH.search(p):
        steps.append({"id": "s1", "action": "web_search", "arg": p})
        steps.append({"id": "s2", "action": "verify_nonempty", "arg": "s1"})
        steps.append({"id": "s3", "action": "compose_answer", "arg": ""})
        return steps

    if _WIKI.search(p):
        topic = re.sub(r"^(who\s+is|what\s+is|wikipedia|explain)\s+", "", p, flags=re.I).strip() or p
        steps.append({"id": "s1", "action": "wikipedia", "arg": topic[:80]})
        steps.append({"id": "s2", "action": "verify_nonempty", "arg": "s1"})
        steps.append({"id": "s3", "action": "compose_answer", "arg": ""})
        return steps

    # default: no tool plan — compose only (optional model)
    steps.append({"id": "s1", "action": "compose_answer", "arg": ""})
    return steps


def _run_tool(action: str, arg: str) -> tuple[bool, str]:
    try:
        if action == "calculator":
            return True, calculate(arg)
        if action == "time":
            return True, now_report()
        if action == "weather":
            return True, weather_report(arg or "local")
        if action == "web_search":
            return True, web_search_report(arg, max_results=4)
        if action == "wikipedia":
            return True, wikipedia_summary(arg)
        if action == "currency":
            parts = arg.split()
            if len(parts) >= 3:
                return True, convert_currency(float(parts[0]), parts[1], parts[2])
            return False, "currency args invalid"
        return False, f"unknown action {action}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def run_si_v1(
    user_text: str,
    generate_fn: Optional[GenerateFn] = None,
    config: Optional[SI_v1Config] = None,
) -> SI_v1Result:
    cfg = config or SI_v1Config()
    t0 = time.perf_counter()
    text = (user_text or "").strip()
    if not text:
        return SI_v1Result(
            answer="(empty input)",
            plan=[],
            stopped_reason="empty_input",
            latency_ms=0.0,
        )

    # Bhutu (metacognition manager): which tools + step budget
    bhutu = bhutu_decide(text)
    if not bhutu.allow:
        return SI_v1Result(
            answer=bhutu.user_message or "Request blocked by Bhutu.",
            plan=[],
            stopped_reason="bhutu_refuse",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
    if bhutu.action == "route_megha":
        try:
            from sentinel.megha import MeghaRequest, run_megha
            mr = run_megha(MeghaRequest(mode=bhutu.route_megha_mode or "checklist", text=text))
            ans = mr.summary or "Megha"
            if mr.items:
                ans += "\n" + "\n".join(
                    ("- " + ("; ".join(f"{k}:{v}" for k, v in it.items()) if isinstance(it, dict) else str(it)))
                    for it in mr.items[:12]
                )
            return SI_v1Result(
                answer=ans,
                plan=[{"id": "megha", "action": "megha", "arg": bhutu.route_megha_mode or ""}],
                stopped_reason="bhutu_megha",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:  # noqa: BLE001
            return SI_v1Result(
                answer=f"Megha route failed: {e}",
                plan=[],
                stopped_reason="bhutu_megha_fail",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    plan = bhutu.filter_plan(build_plan(text))
    if bhutu.action == "no_tool":
        plan = [s for s in plan if str(s.get("action")) == "compose_answer"] or [
            {"id": "s1", "action": "compose_answer", "arg": ""}
        ]
    # respect Bhutu step budget
    cfg = SI_v1Config(
        max_steps=min(cfg.max_steps, max(1, bhutu.max_agent_steps)),
        max_failures=cfg.max_failures,
        tool_timeout_s=cfg.tool_timeout_s,
    )
    results: dict[str, str] = {}
    step_logs: list[StepResult] = []
    facts: list[str] = []
    failures = 0
    n_run = 0

    for spec in plan:
        if n_run >= cfg.max_steps:
            return SI_v1Result(
                answer=facts[-1] if facts else "(max steps)",
                plan=plan,
                steps=step_logs,
                stopped_reason="max_steps",
                latency_ms=(time.perf_counter() - t0) * 1000,
                facts=facts,
            )
        n_run += 1
        action = spec["action"]
        arg = spec.get("arg") or ""
        sid = spec["id"]

        if action == "verify_nonempty":
            ref = results.get(arg, "")
            ok = bool(ref.strip()) and not ref.lower().startswith("error")
            ts = time.perf_counter()
            step_logs.append(
                StepResult(sid, action, ok, "ok" if ok else f"missing/fail ref={arg}", (time.perf_counter() - ts) * 1000)
            )
            if not ok:
                failures += 1
                if failures >= cfg.max_failures:
                    return SI_v1Result(
                        answer="Could not verify tool results.",
                        plan=plan,
                        steps=step_logs,
                        stopped_reason="verify_failed",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        facts=facts,
                    )
            continue

        if action == "compose_answer":
            ts = time.perf_counter()
            body = "\n".join(facts) if facts else ""
            if generate_fn is not None:
                prompt = (
                    f"[Verified tool facts]\n{body}\n\nUser: {text}\n"
                    "Answer briefly using only the facts when relevant."
                )
                try:
                    gen = (generate_fn(prompt) or "").strip()
                except Exception as e:  # noqa: BLE001
                    gen = ""
                    step_logs.append(
                        StepResult(sid, "generate", False, str(e), (time.perf_counter() - ts) * 1000)
                    )
                if gen:
                    answer = gen
                    step_logs.append(
                        StepResult(sid, "compose_answer", True, "model", (time.perf_counter() - ts) * 1000)
                    )
                    return SI_v1Result(
                        answer=answer,
                        plan=plan,
                        steps=step_logs,
                        stopped_reason="answered_model",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        facts=facts,
                    )
            answer = body if body else (
                "No tool matched. Provide a calc/time/weather/search question, "
                "or attach generate_fn / train SFT for open chat."
            )
            step_logs.append(
                StepResult(sid, "compose_answer", True, "facts_only", (time.perf_counter() - ts) * 1000)
            )
            return SI_v1Result(
                answer=answer,
                plan=plan,
                steps=step_logs,
                stopped_reason="answered_facts" if body else "no_tools",
                latency_ms=(time.perf_counter() - t0) * 1000,
                facts=facts,
            )

        # tool step
        ts = time.perf_counter()
        if not bhutu.may_run_tool(action):
            step_logs.append(
                StepResult(sid, action, False, "bhutu_denied", 0.0)
            )
            failures += 1
            continue
        ok, detail = _run_tool(action, arg)
        ms = (time.perf_counter() - ts) * 1000
        if ms > cfg.tool_timeout_s * 1000:
            ok = False
            detail = f"timeout ({ms:.0f}ms)"
        step_logs.append(StepResult(sid, action, ok, detail[:2000], ms))
        results[sid] = detail
        if ok:
            facts.append(detail)
        else:
            failures += 1
            if failures >= cfg.max_failures:
                return SI_v1Result(
                    answer=f"Tool failed: {detail[:300]}",
                    plan=plan,
                    steps=step_logs,
                    stopped_reason="tool_fail_budget",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    facts=facts,
                )

    return SI_v1Result(
        answer=facts[-1] if facts else "(no answer)",
        plan=plan,
        steps=step_logs,
        stopped_reason="plan_exhausted",
        latency_ms=(time.perf_counter() - t0) * 1000,
        facts=facts,
    )
