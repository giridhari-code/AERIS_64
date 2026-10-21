"""0z code runtime — full company agent loop (plan → tools → synthesize)."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from sentinel.agent.loop import AgentResult, ToolEvent
from sentinel.zeroz.config import ZeroZConfig
from sentinel.zeroz.planner import Plan, build_plan
from sentinel.zeroz.policy import (
    PRODUCT_NAME,
    SYSTEM_PROMPT,
    check_user_message,
)
from sentinel.zeroz.session import ZeroZSession
from sentinel.zeroz.toolbus import ToolBus

logger = logging.getLogger("zeroz")

GenerateFn = Callable[[str], str]


def _wrap_generate(generate_fn: Optional[GenerateFn]) -> Optional[GenerateFn]:
    if generate_fn is None:
        return None

    def _gen(prompt: str) -> str:
        full = f"{SYSTEM_PROMPT}\n\n---\n{prompt}"
        text = generate_fn(full)
        if "```" in (text or "") and any(
            x in text.lower() for x in ("def ", "function ", "import ", "const ", "class ")
        ):
            return (
                f"{PRODUCT_NAME}: code-like model output suppressed.\n"
                "Ask for a non-code plan, checklist, or analysis instead."
            )
        return text

    return _gen


def _synthesize(plan: Plan, tool_notes: list[str], generate_fn: Optional[GenerateFn]) -> str:
    facts = "\n\n".join(tool_notes) if tool_notes else ""
    if generate_fn is not None:
        prompt = (
            f"User objective: {plan.objective}\n\n"
            f"Tool facts (trust these; do not invent numbers):\n{facts or '(none)'}\n\n"
            "Write a clear professional answer. No source code."
        )
        try:
            text = (generate_fn(prompt) or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("generate failed: %s", e)
    if facts:
        return f"**{PRODUCT_NAME} — results**\n\n{facts}"
    return (
        f"{PRODUCT_NAME}: no tools matched and no model reply. "
        "Try weather, search, time, calc, wiki, git status, or file search."
    )


def run_zeroz(
    user_message: str,
    *,
    cfg: ZeroZConfig | None = None,
    session: ZeroZSession | None = None,
    generate_fn: Optional[GenerateFn] = None,
) -> AgentResult:
    """Full 0z turn: policy → plan → toolbus → optional LM synthesize."""
    cfg = cfg or ZeroZConfig()
    session = session or ZeroZSession()
    t0 = time.perf_counter()

    decision = check_user_message(user_message)
    session.add("user", user_message)
    session.log("user_message", text=user_message[:2000])

    if not decision.allow:
        session.add("assistant", decision.public_message, refused=True)
        session.log("policy_refuse", reason=decision.reason)
        return AgentResult(
            answer=decision.public_message,
            stopped_reason="policy_refuse",
            steps=0,
            tool_events=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    bus = ToolBus(allowed=set(cfg.allowed_tools))
    plan = build_plan(user_message)
    session.log("plan", plan=plan.to_dict())

    events: list[ToolEvent] = []
    notes: list[str] = []
    failures = 0
    steps = 0

    for step in plan.steps[: cfg.max_steps]:
        steps += 1
        if step.tool is None:
            continue
        result = bus.call(step.tool, step.args)
        events.append(
            ToolEvent(result.name, result.ok, result.output[:2000], result.ms)
        )
        if result.ok:
            notes.append(f"[{result.name}] {result.output}")
        else:
            failures += 1
            notes.append(f"[{result.name} FAILED] {result.output}")
            if failures >= cfg.max_tool_failures:
                break

    gen = _wrap_generate(generate_fn) if cfg.use_model_generate else None
    # Optional context
    if session.turns and len(session.turns) > 1:
        # already logged user; transcript for model only
        pass
    answer = _synthesize(plan, notes, gen)
    steps += 1

    reason = "answered"
    if not notes and not plan.steps:
        reason = "no_tools"
    if failures >= cfg.max_tool_failures and not any(e.ok for e in events):
        reason = "tool_fail_budget"

    result = AgentResult(
        answer=answer,
        stopped_reason=reason,
        steps=steps,
        tool_events=events,
        facts_block="\n\n".join(notes) if notes else None,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    session.add("assistant", result.answer, stopped=result.stopped_reason)
    session.log("turn_complete", stopped=result.stopped_reason, metrics=result.metrics())
    if cfg.audit_dir:
        try:
            session.save_audit(cfg.audit_dir)
        except OSError as e:
            logger.warning("audit save failed: %s", e)
    return result


def build_generate_fn_from_checkpoint(
    checkpoint: str,
    device: str = "cpu",
    max_new_tokens: int = 128,
) -> Optional[GenerateFn]:
    try:
        import torch
        from sentinel.checkpoint.resolve import load_model_from_checkpoint
    except Exception as e:
        logger.warning("checkpoint load unavailable: %s", e)
        return None
    try:
        model, tok, cfg, meta = load_model_from_checkpoint(checkpoint, None, device=device)
        model.eval()
    except Exception as e:
        logger.warning("failed to load checkpoint %s: %s", checkpoint, e)
        return None

    def _generate(prompt: str) -> str:
        ids: list[int] = []
        if tok is not None and hasattr(tok, "encode"):
            try:
                enc = tok.encode(prompt)
                ids = list(enc.ids) if hasattr(enc, "ids") else list(enc)
            except Exception:
                ids = [ord(c) % 32000 for c in prompt[-512:]]
        else:
            ids = [ord(c) % 32000 for c in prompt[-512:]]
        if not ids:
            return ""
        device_t = next(model.parameters()).device
        x = torch.tensor([ids[-256:]], device=device_t)
        out_ids = list(ids[-256:])
        state = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                o = model(x[:, -1:], state=state) if state is not None else model(x)
                logits = getattr(o, "logits", o)
                state = getattr(o, "state", None)
                nxt = int(logits[0, -1].argmax().item())
                out_ids.append(nxt)
                x = torch.tensor([[nxt]], device=device_t)
                if nxt == 0:
                    break
        if tok is not None and hasattr(tok, "decode"):
            try:
                return tok.decode(out_ids[len(ids[-256:]) :])
            except Exception:
                pass
        return " ".join(str(i) for i in out_ids[-32:])

    return _generate
