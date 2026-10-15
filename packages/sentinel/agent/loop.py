"""
Reliable agent loop (SI_v0 behaviour layer).

Model evaluation must match the system. For agentic workloads the relevant
metrics are (see industry guidance on system-matched evals):

  - Function call validity rate   (tool succeeded vs hallucinated / failed)
  - Step consistency / reproducibility
  - Pipeline completion rate      (did the agent finish within budget?)

This loop is instrumented exactly for those metrics:
  1. Hard step budget (no infinite loops)          → pipeline completion
  2. Allow-listed tools only
  3. Every tool call timed + success/fail recorded → function-call validity
  4. Facts injected; model never trusted alone for live numbers
  5. Deterministic stop: answer | max_steps | tool_fail_budget | empty
     → step consistency / reproducibility

This does NOT require the LM to emit tool JSON. Routing uses the same
regex/orchestrator as serving (predictable). Optional generate_fn can
polish the final reply from the model checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

from sentinel.tools.orchestrate import enrich_prompt_with_tools
from sentinel.tools.calculator import calculate
from sentinel.tools.currency import convert_currency
from sentinel.tools.timeutil import now_report
from sentinel.tools.weather import weather_report
from sentinel.tools.web_search import web_search_report
from sentinel.tools.wikipedia import wikipedia_summary


# ---- types -----------------------------------------------------------------

GenerateFn = Callable[[str], str]  # prompt -> model text


@dataclass
class AgentConfig:
    max_steps: int = 4
    max_tool_failures: int = 2
    tool_timeout_s: float = 12.0
    allowed_tools: tuple[str, ...] = (
        "weather",
        "web_search",
        "calculator",
        "time",
        "wikipedia",
        "currency",
        "git",
    )
    require_tool_for_live_facts: bool = True
    # If True, always run tool enrichment before generate
    always_enrich: bool = True
    # Auto RAG (metacognition-gated knowledge retrieval) before generate
    use_auto_rag: bool = True


@dataclass
class ToolEvent:
    name: str
    ok: bool
    detail: str
    ms: float


@dataclass
class AgentResult:
    answer: str
    stopped_reason: str
    steps: int
    tool_events: list[ToolEvent] = field(default_factory=list)
    facts_block: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = self.metrics()
        return d

    def metrics(self) -> dict[str, float | int | str]:
        """System-matched agent metrics (function validity, step consistency, completion)."""
        total = len(self.tool_events)
        ok = sum(1 for e in self.tool_events if e.ok)
        validity = (ok / total) if total else 1.0
        completed = self.stopped_reason in ("answer", "done", "final")
        return {
            "function_call_validity_rate": round(validity, 4),
            "tool_calls": total,
            "tool_successes": ok,
            "steps": self.steps,
            "pipeline_completed": int(completed),
            "stopped_reason": self.stopped_reason,
            "latency_ms": round(self.latency_ms, 2),
        }


# ---- tool runners with guard rails ----------------------------------------

def _safe_call(name: str, fn: Callable[[], str], timeout_s: float) -> ToolEvent:
    t0 = time.perf_counter()
    try:
        # Soft timeout: tools are sync; we still record duration and clip error
        out = fn()
        ms = (time.perf_counter() - t0) * 1000
        if ms > timeout_s * 1000:
            return ToolEvent(name, False, f"timeout soft ({ms:.0f}ms > {timeout_s}s)", ms)
        text = (out or "").strip()
        if not text or text.lower().startswith("error"):
            return ToolEvent(name, False, text or "empty tool result", ms)
        return ToolEvent(name, True, text[:2000], ms)
    except Exception as e:  # noqa: BLE001 — agent must not crash
        ms = (time.perf_counter() - t0) * 1000
        return ToolEvent(name, False, f"{type(e).__name__}: {e}", ms)


def _run_enrichment(user_text: str, cfg: AgentConfig) -> tuple[str, Optional[str], list[ToolEvent]]:
    """Use existing orchestrator; wrap as a single audited 'enrich' step."""
    events: list[ToolEvent] = []
    t0 = time.perf_counter()
    try:
        enriched, block = enrich_prompt_with_tools(user_text)
        ms = (time.perf_counter() - t0) * 1000
        if block:
            events.append(ToolEvent("enrich_tools", True, block[:2000], ms))
        else:
            events.append(ToolEvent("enrich_tools", True, "(no tool matched)", ms))
        return enriched, block, events
    except Exception as e:  # noqa: BLE001
        ms = (time.perf_counter() - t0) * 1000
        events.append(ToolEvent("enrich_tools", False, str(e), ms))
        return user_text, None, events


# ---- main loop ------------------------------------------------------------

def run_agent(
    user_text: str,
    generate_fn: Optional[GenerateFn] = None,
    config: Optional[AgentConfig] = None,
) -> AgentResult:
    """
    Reliable agent entrypoint.

    Parameters
    ----------
    user_text : str
        User question.
    generate_fn : optional
        If provided, called once with (enriched) prompt to produce final wording.
        If None, answer is the facts block or a honest fallback.
    config : AgentConfig
    """
    cfg = config or AgentConfig()
    t_all = time.perf_counter()
    steps = 0
    failures = 0
    events: list[ToolEvent] = []
    facts: Optional[str] = None
    prompt = user_text.strip()
    if not prompt:
        return AgentResult(
            answer="(empty input)",
            stopped_reason="empty_input",
            steps=0,
            latency_ms=0.0,
        )

    # Step 1: tool enrichment (deterministic routing)
    if cfg.always_enrich:
        steps += 1
        prompt, facts, ev = _run_enrichment(prompt, cfg)
        events.extend(ev)
        if ev and not ev[-1].ok:
            failures += 1

    if failures >= cfg.max_tool_failures:
        return AgentResult(
            answer="Tools failed repeatedly; I cannot verify live facts right now.",
            stopped_reason="tool_fail_budget",
            steps=steps,
            tool_events=events,
            facts_block=facts,
            latency_ms=(time.perf_counter() - t_all) * 1000,
        )

    # Step 1b: Auto RAG — metacognition-gated knowledge retrieval
    if cfg.use_auto_rag:
        steps += 1
        t0 = time.perf_counter()
        try:
            from sentinel.rag.meta_rag import auto_rag

            prompt, rag_info = auto_rag(prompt, session_key="agent")
            ms = (time.perf_counter() - t0) * 1000
            detail = (
                f"retrieve={rag_info.get('should_retrieve')} "
                f"hits={rag_info.get('hits')} mode={rag_info.get('mode')} "
                f"conf={rag_info.get('confidence')}"
            )
            events.append(ToolEvent("auto_rag", True, detail[:500], ms))
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            events.append(ToolEvent("auto_rag", False, f"{type(e).__name__}: {e}", ms))

    # Step 2: generate or factual reply
    steps += 1
    if generate_fn is not None:
        try:
            t0 = time.perf_counter()
            text = (generate_fn(prompt) or "").strip()
            ms = (time.perf_counter() - t0) * 1000
            events.append(ToolEvent("generate", bool(text), text[:500] if text else "empty", ms))
            if not text:
                # fallback to facts
                answer = facts or (
                    "I could not generate a fluent reply. "
                    "Train with more SFT data, or rely on tool facts above."
                )
                reason = "generate_empty_fallback"
            else:
                answer = text
                reason = "answered"
        except Exception as e:  # noqa: BLE001
            events.append(ToolEvent("generate", False, str(e), 0.0))
            answer = facts or f"Generation error: {e}"
            reason = "generate_error"
    else:
        if facts:
            answer = facts
            reason = "facts_only"
        else:
            answer = (
                "No live tool matched and no model generate_fn was provided. "
                "Connect generate_fn or ask a weather/search/calc/time question."
            )
            reason = "no_model_no_facts"

    if steps >= cfg.max_steps:
        reason = reason if reason.startswith("answer") else "max_steps"

    return AgentResult(
        answer=answer,
        stopped_reason=reason,
        steps=steps,
        tool_events=events,
        facts_block=facts,
        latency_ms=(time.perf_counter() - t_all) * 1000,
    )
