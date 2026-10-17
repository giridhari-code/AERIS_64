"""KALO BOT mother agent — orchestrates children; does not invent tool results.

Production path:
  1. Route intent (router)
  2. Bhutu gate (policy) when tools may run
  3. Dispatch to buddy / megha / dataset_builder / tool enrich / plain chat
  4. Return structured KaloTurn (answer + route + events)

The language model generate_fn is injected by the server so this module stays
free of hard-coded replies and checkpoint paths.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from sentinel.kalo.identity import MODEL_NAME, PRODUCT_NAME, TAGLINE, system_preamble
from sentinel.kalo.router import Route, route_intent

GenerateFn = Callable[[str], str]


@dataclass
class KaloEvent:
    name: str
    ok: bool
    detail: str
    ms: float = 0.0


@dataclass
class KaloTurn:
    answer: str
    route: str
    route_reason: str
    confidence: float
    events: list[KaloEvent] = field(default_factory=list)
    facts: Optional[str] = None
    latency_ms: float = 0.0
    product: str = PRODUCT_NAME
    tagline: str = TAGLINE
    model: str = MODEL_NAME

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


def _timed(name: str, fn: Callable[[], str]) -> KaloEvent:
    t0 = time.perf_counter()
    try:
        out = fn()
        return KaloEvent(name, True, (out or "")[:4000], (time.perf_counter() - t0) * 1000)
    except Exception as e:
        return KaloEvent(name, False, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000)


class KaloBot:
    """Mother agent. One instance can serve many turns (stateless per call)."""

    def __init__(self, generate: Optional[GenerateFn] = None) -> None:
        self._generate = generate

    def set_generate(self, generate: GenerateFn) -> None:
        self._generate = generate

    def run(
        self,
        prompt: str,
        *,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        generate: Optional[GenerateFn] = None,
    ) -> KaloTurn:
        t0 = time.perf_counter()
        text = (prompt or "").strip()
        route = route_intent(text)
        events: list[KaloEvent] = []
        facts: Optional[str] = None
        answer = ""

        gen = generate or self._generate

        # --- Buddy ----------------------------------------------------------
        if route.target == "buddy":
            def _buddy() -> str:
                from sentinel.buddy import handle_buddy_command, is_buddy_command
                from sentinel.buddy.eggs import EggStore
                import os
                from pathlib import Path

                egg = None
                try:
                    store = EggStore(
                        Path(os.environ.get("NEUROFIELD_BUDDY_EGG_PATH") or "artifacts/buddy/eggs.json")
                    )
                    egg = store.snapshot(user_id)
                except Exception:
                    egg = None
                if is_buddy_command(text):
                    out = handle_buddy_command(text, user_id, egg=egg)
                    return out or "Buddy: no reply"
                return handle_buddy_command("/buddy status", user_id, egg=egg) or "Buddy status"

            ev = _timed("buddy", _buddy)
            events.append(ev)
            answer = ev.detail if ev.ok else f"Buddy error: {ev.detail}"

        # --- Megha ----------------------------------------------------------
        elif route.target == "megha":
            def _megha() -> str:
                from sentinel.tools.megha_tool import megha_report

                return megha_report(text)

            ev = _timed("megha", _megha)
            events.append(ev)
            answer = ev.detail if ev.ok else f"Megha error: {ev.detail}"

        # --- Dataset Builder ------------------------------------------------
        elif route.target == "dataset":
            def _ds() -> str:
                from sentinel.tools.dataset_builder import dataset_builder_report

                # map rough verbs
                low = text.lower()
                if "export" in low and "train" in low:
                    return dataset_builder_report("export_train")
                if "export" in low:
                    return dataset_builder_report("export")
                if "balance" in low:
                    return dataset_builder_report("balance")
                if "imbalance" in low:
                    return dataset_builder_report("imbalance")
                return dataset_builder_report("progress")

            ev = _timed("dataset_builder", _ds)
            events.append(ev)
            answer = ev.detail if ev.ok else f"DatasetBuilder error: {ev.detail}"

        # --- Tools (+ Bhutu gate) -------------------------------------------
        elif route.target in {"tools", "bhutu", "ajeebwork", "kalofacts", "chat"}:
            # Bhutu pre-check for toolish routes
            if route.target in {"tools", "bhutu"}:
                def _bhutu() -> str:
                    from sentinel.modules.bhutu import bhutu_decide

                    d = bhutu_decide(text)
                    allowed = getattr(d, "allowed_tools", None) or getattr(d, "tools", None) or []
                    refuse = getattr(d, "refuse", False) or getattr(d, "block", False)
                    note = getattr(d, "reason", None) or getattr(d, "message", "") or ""
                    return f"refuse={refuse} allowed={list(allowed)[:12]} {note}"

                events.append(_timed("bhutu", _bhutu))

            if route.target == "tools" or route.target == "chat":
                def _enrich() -> str:
                    from sentinel.tools.orchestrate import enrich_prompt_with_tools

                    enriched, note = enrich_prompt_with_tools(text)
                    return note or ""

                ev = _timed("tools_enrich", _enrich)
                events.append(ev)
                if ev.ok and ev.detail:
                    facts = ev.detail

            # Generate via injected model
            if gen is not None:
                def _gen() -> str:
                    block = system_preamble()
                    user = text
                    if facts:
                        user = f"{text}\n\n[tool_facts]\n{facts}\n[/tool_facts]"
                    return gen(f"{block}\n\nUser: {user}\n\nAssistant:")

                gev = _timed("generate", _gen)
                events.append(gev)
                answer = gev.detail if gev.ok else f"Generate error: {gev.detail}"
            else:
                # No model bound — still return useful child/tool output
                if facts:
                    answer = facts
                elif events:
                    answer = events[-1].detail
                else:
                    answer = (
                        f"{PRODUCT_NAME} routed to `{route.target}` ({route.reason}) "
                        "but no generate_fn is bound. Start the AERIS_64 backend with a checkpoint."
                    )

            if route.target == "ajeebwork":
                events.append(
                    KaloEvent("ajeebwork", True, "AjeebWork dispatch is available via tools/ajeebwork; use plan in chat.", 0)
                )
            if route.target == "kalofacts":
                events.append(
                    KaloEvent("kalofacts", True, "Kalofacts API: /v1/kalofacts", 0)
                )

        else:
            answer = f"Unhandled route: {route.target}"

        return KaloTurn(
            answer=(answer or "").strip() or "(empty)",
            route=route.target,
            route_reason=route.reason,
            confidence=route.confidence,
            events=events,
            facts=facts,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def run_kalo(
    prompt: str,
    *,
    user_id: str = "anonymous",
    generate: Optional[GenerateFn] = None,
) -> dict[str, Any]:
    return KaloBot(generate=generate).run(prompt, user_id=user_id).to_dict()
