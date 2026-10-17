"""KALO BOT mother agent — production orchestrator.

Dispatches to real subsystems only. No canned product demos as answers.
Generate is injected by the serving layer when a checkpoint is loaded.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from sentinel.kalo.identity import MODEL_NAME, PRODUCT_NAME, TAGLINE, system_preamble
from sentinel.kalo.router import route_intent

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
        return KaloEvent(name, True, (out or "")[:8000], (time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        return KaloEvent(
            name, False, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000.0
        )


def _egg_snapshot(user_id: str) -> Optional[dict[str, Any]]:
    try:
        from sentinel.buddy.eggs import EggStore

        path = Path(
            os.environ.get("NEUROFIELD_BUDDY_EGG_PATH")
            or os.environ.get("AERIS_BUDDY_EGG_PATH")
            or "artifacts/buddy/eggs.json"
        )
        return EggStore(path).snapshot(user_id)
    except Exception:
        return None


def _dataset_action(text: str) -> str:
    low = text.lower()
    if re_search(low, r"export\s*train|training\s*jsonl|export_train"):
        return "export_train"
    if "export" in low:
        return "export"
    if "balance" in low and "imbalance" not in low:
        return "balance"
    if "imbalance" in low:
        return "imbalance"
    if re_search(low, r"collect|fetch|url|ingest|add\s+file"):
        return "help"
    return "progress"


def re_search(text: str, pat: str) -> bool:
    import re

    return bool(re.search(pat, text, re.I))


class KaloBot:
    def __init__(self, generate: Optional[GenerateFn] = None) -> None:
        self._generate = generate

    def set_generate(self, generate: GenerateFn) -> None:
        self._generate = generate

    def run(
        self,
        prompt: str,
        *,
        user_id: str = "anonymous",
        generate: Optional[GenerateFn] = None,
    ) -> KaloTurn:
        t0 = time.perf_counter()
        text = (prompt or "").strip()
        route = route_intent(text)
        events: list[KaloEvent] = []
        facts: Optional[str] = None
        answer = ""
        gen = generate or self._generate

        if route.target == "buddy":
            def _buddy() -> str:
                from sentinel.buddy import handle_buddy_command, is_buddy_command, resolve_buddy_reply

                egg = _egg_snapshot(user_id)
                cmd = text if is_buddy_command(text) else "/buddy status"
                raw = handle_buddy_command(cmd, user_id, egg=egg) or ""
                msg, _meta = resolve_buddy_reply(raw)
                return msg or raw or "Buddy: no reply"

            ev = _timed("buddy", _buddy)
            events.append(ev)
            answer = ev.detail if ev.ok else f"Buddy error: {ev.detail}"

        elif route.target == "megha":
            def _megha() -> str:
                from sentinel.tools.megha_tool import megha_report

                return megha_report(text)

            ev = _timed("megha", _megha)
            events.append(ev)
            answer = ev.detail if ev.ok else f"Megha error: {ev.detail}"

        elif route.target == "dataset":
            def _ds() -> str:
                from sentinel.tools.dataset_builder import dataset_builder_report

                return dataset_builder_report(_dataset_action(text))

            ev = _timed("dataset_builder", _ds)
            events.append(ev)
            answer = ev.detail if ev.ok else f"DatasetBuilder error: {ev.detail}"

        elif route.target == "kalofacts":
            def _kf() -> str:
                try:
                    from sentinel.kalofacts.store import list_artifacts

                    items = list_artifacts() if callable(list_artifacts) else []
                    n = len(items) if items is not None else 0
                    return f"Kalofacts: {n} artifact(s). API: GET/POST /v1/kalofacts"
                except Exception as e:
                    return f"Kalofacts API: /v1/kalofacts ({type(e).__name__})"

            ev = _timed("kalofacts", _kf)
            events.append(ev)
            answer = ev.detail

        elif route.target == "ajeebwork":
            def _aw() -> str:
                try:
                    from sentinel.tools import ajeebwork_tools  # noqa: F401

                    return "AjeebWork tools registered. Drive multi-step file work via agent tools catalog."
                except Exception as e:
                    return f"AjeebWork unavailable: {type(e).__name__}: {e}"

            ev = _timed("ajeebwork", _aw)
            events.append(ev)
            answer = ev.detail

        else:
            # tools | bhutu | chat
            allow_tools = True
            if route.target in {"tools", "bhutu", "chat"}:
                def _bhutu() -> str:
                    from sentinel.modules.bhutu import bhutu_decide

                    d = bhutu_decide(text)
                    allow = bool(getattr(d, "allow", True))
                    msg = getattr(d, "user_message", "") or getattr(d, "reason", "") or ""
                    tools = getattr(d, "allowed_tools", None) or []
                    return f"allow={allow} tools={list(tools)[:16]} {msg}".strip()

                bev = _timed("bhutu", _bhutu)
                events.append(bev)
                if bev.ok and bev.detail.startswith("allow=False"):
                    allow_tools = False
                    answer = bev.detail

            if allow_tools and route.target in {"tools", "chat"}:
                def _enrich() -> str:
                    from sentinel.tools.orchestrate import enrich_prompt_with_tools

                    _enriched, note = enrich_prompt_with_tools(text)
                    return (note or "").strip()

                eev = _timed("tools_enrich", _enrich)
                events.append(eev)
                if eev.ok and eev.detail:
                    facts = eev.detail

            if not answer:
                if gen is not None:
                    def _gen() -> str:
                        user = text
                        if facts:
                            user = f"{text}\n\n[tool_facts]\n{facts}\n[/tool_facts]"
                        return gen(f"{system_preamble()}\n\nUser: {user}\n\nAssistant:")

                    gev = _timed("generate", _gen)
                    events.append(gev)
                    answer = gev.detail if gev.ok else f"Generate error: {gev.detail}"
                elif facts:
                    answer = facts
                else:
                    answer = (
                        f"{PRODUCT_NAME} routed to `{route.target}` ({route.reason}). "
                        "No model checkpoint bound — load NEUROFIELD_CHECKPOINT for generation, "
                        "or ask a tool/buddy/dataset command."
                    )

        return KaloTurn(
            answer=(answer or "").strip() or "(empty)",
            route=route.target,
            route_reason=route.reason,
            confidence=route.confidence,
            events=events,
            facts=facts,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


def run_kalo(
    prompt: str,
    *,
    user_id: str = "anonymous",
    generate: Optional[GenerateFn] = None,
) -> dict[str, Any]:
    return KaloBot(generate=generate).run(prompt, user_id=user_id).to_dict()
