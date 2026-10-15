"""Bhutu — Metacognition tool/agent manager for SENTINEL.

Sits under the metacognition control layer and decides, for each user turn:

  MONITOR  — confidence / arousal / conflict / Megha policy
  CONTROL  — refuse | no_tool | which tools | agent max_steps | RAG | budgets

Nothing here executes network attacks. Illegal cyber intents are refused via
Megha policy. Sandbox/shell stays allow-listed and only opens when Bhutu says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence


# Tool families Bhutu may authorize (names match SI_v1 / catalog / common aliases)
SAFE_FACT_TOOLS = frozenset({
    "calculator", "calc", "time", "weather", "web_search", "search",
    "wikipedia", "wiki", "currency", "web_fetch",
})
RAG_TOOLS = frozenset({"rag", "file_search", "retrieve"})
WORKSPACE_TOOLS = frozenset({
    "read_file", "write_file", "edit_file", "git", "ajeebwork", "kalofacts",
})
SANDBOX_TOOLS = frozenset({
    "bash", "shell", "code_execution", "python", "computer_use",
})
ALL_KNOWN = SAFE_FACT_TOOLS | RAG_TOOLS | WORKSPACE_TOOLS | SANDBOX_TOOLS | frozenset({
    "tool_search", "compose_answer", "megha",
})


@dataclass
class BhutuDecision:
    """Manager output for one user turn."""

    allow: bool
    action: str  # refuse | no_tool | use_tools | route_megha
    reason: str
    allowed_tools: list[str] = field(default_factory=list)
    max_agent_steps: int = 4
    use_rag: bool = False
    use_tool_search: bool = False
    allow_sandbox: bool = False
    route_megha_mode: Optional[str] = None
    confidence: float = 0.5
    arousal: float = 0.0
    conflict: float = 0.0
    user_message: str = ""  # set when refuse
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def filter_plan(self, steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only plan steps whose action Bhutu authorized."""
        if not self.allow:
            return []
        if self.action == "no_tool":
            return [s for s in steps if str(s.get("action") or "") in {"compose_answer", ""}]
        allowed = set(self.allowed_tools) | {"compose_answer"}
        # aliases
        if "calculator" in allowed:
            allowed.add("calc")
        if "wikipedia" in allowed:
            allowed.add("wiki")
        if "web_search" in allowed:
            allowed.add("search")
        out: list[dict[str, Any]] = []
        for s in steps:
            act = str(s.get("action") or "").strip().lower()
            if act in allowed or act == "compose_answer":
                out.append(s)
        return out

    def may_run_tool(self, name: str) -> bool:
        if not self.allow or self.action in {"refuse", "no_tool"}:
            return False
        n = (name or "").strip().lower()
        if n in {"compose_answer", ""}:
            return True
        if n in SANDBOX_TOOLS and not self.allow_sandbox:
            return False
        return n in set(self.allowed_tools) or n in {
            a for a in self.allowed_tools
        }


class Bhutu:
    """Metacognition manager: tools + agent + tool_search + RAG budget."""

    name = "Bhutu"

    def __init__(
        self,
        *,
        max_steps_cap: int = 6,
        default_steps: int = 4,
        enable_sandbox: bool = False,
    ):
        self.max_steps_cap = max(1, int(max_steps_cap))
        self.default_steps = max(1, min(self.max_steps_cap, int(default_steps)))
        self.enable_sandbox = bool(enable_sandbox)

    def decide(
        self,
        prompt: str,
        *,
        confidence: Optional[float] = None,
        arousal: Optional[float] = None,
        conflict: Optional[float] = None,
        authorized: bool = False,
        scope: str = "",
    ) -> BhutuDecision:
        text = (prompt or "").strip()
        conf, arou, confct, heur_reason = self._estimate(text)
        if confidence is not None:
            conf = float(confidence)
        if arousal is not None:
            arou = float(arousal)
        if conflict is not None:
            confct = float(conflict)

        # 1) Policy (Megha) — hard refuse
        try:
            from sentinel.megha.policy import evaluate_user_text
            pol = evaluate_user_text(text, authorized=authorized, scope=scope)
        except Exception as e:  # noqa: BLE001
            pol = None
            pol_err = str(e)
        else:
            pol_err = None

        if pol is not None and not pol.allow:
            return BhutuDecision(
                allow=False,
                action="refuse",
                reason=pol.reason or "policy_refuse",
                user_message=pol.user_message or "Blocked by Bhutu (Megha policy).",
                confidence=conf,
                arousal=arou,
                conflict=confct,
                max_agent_steps=0,
                diagnostics={"policy": pol.to_dict(), "heuristic": heur_reason},
            )

        if pol is not None and pol.action == "route_megha":
            return BhutuDecision(
                allow=True,
                action="route_megha",
                reason=pol.reason or "defensive_route",
                route_megha_mode=pol.megha_mode or "checklist",
                allowed_tools=["megha"],
                max_agent_steps=1,
                confidence=conf,
                arousal=arou,
                conflict=confct,
                diagnostics={"policy": pol.to_dict(), "heuristic": heur_reason},
            )

        # 2) Intent → tool families
        tools, use_rag, want_search, want_sandbox = self._select_tools(text, conf, arou, confct)
        if want_sandbox and not self.enable_sandbox:
            want_sandbox = False

        if not tools and not use_rag:
            return BhutuDecision(
                allow=True,
                action="no_tool",
                reason=f"compose_only:{heur_reason}",
                allowed_tools=[],
                max_agent_steps=1,
                use_rag=False,
                use_tool_search=False,
                allow_sandbox=False,
                confidence=conf,
                arousal=arou,
                conflict=confct,
                diagnostics={"heuristic": heur_reason},
            )

        # 3) Budget: low confidence / high arousal → more steps
        steps = self.default_steps
        if conf < 0.4 or arou > 1.0 or confct > 0.8:
            steps = min(self.max_steps_cap, steps + 2)
        if conf > 0.85 and arou < 0.3:
            steps = max(1, min(steps, 2))

        return BhutuDecision(
            allow=True,
            action="use_tools",
            reason=f"tools:{heur_reason}",
            allowed_tools=sorted(tools),
            max_agent_steps=steps,
            use_rag=use_rag,
            use_tool_search=want_search,
            allow_sandbox=want_sandbox,
            confidence=conf,
            arousal=arou,
            conflict=confct,
            diagnostics={"heuristic": heur_reason, "pol_err": pol_err},
        )

    def _estimate(self, text: str) -> tuple[float, float, float, str]:
        """Prompt heuristic aligned with meta_rag ranges when live meta signals absent."""
        if not text:
            return 0.7, 0.0, 0.0, "empty"
        low = text.lower()
        conf, arou, conflict = 0.65, 0.2, 0.2
        reason = "neutral"
        if re.search(r"\b(what|why|how|when|where|who|kya|kaise|kab|kyun)\b", low):
            conf -= 0.1
            arou += 0.2
            reason = "question"
        if re.search(r"\b(not sure|pata nahi|maybe|shayad)\b", low):
            conf -= 0.2
            conflict += 0.3
            reason = "uncertainty"
        if re.search(r"\b(calculate|compute|\d+\s*[\+\-\*/]\s*\d+)\b", low):
            conf = max(conf, 0.7)
            reason = "calc"
        if re.search(r"\b(weather|temperature|rain)\b", low):
            reason = "weather"
            conf = 0.55
        if re.search(r"\b(search|latest|news|who is|wikipedia)\b", low):
            conf = min(conf, 0.45)
            arou += 0.3
            reason = "factual_search"
        if len(text) > 400:
            conflict += 0.15
        conf = max(0.05, min(0.95, conf))
        arou = max(0.0, arou)
        conflict = max(0.0, conflict)
        return conf, arou, conflict, reason

    def _select_tools(
        self, text: str, conf: float, arou: float, conflict: float
    ) -> tuple[set[str], bool, bool, bool]:
        low = text.lower()
        tools: set[str] = set()
        use_rag = False
        want_search = False
        want_sandbox = False

        if re.search(r"(\d+\s*[\+\-\*/]\s*\d+)|calculate|compute|percent", low):
            tools.add("calculator")
        if re.search(r"\b(time|clock|date|aaj\s+kitne|what time)\b", low):
            tools.add("time")
        if re.search(r"\b(weather|temperature|forecast|barish|mausam)\b", low):
            tools.add("weather")
        if re.search(r"\b(wiki|wikipedia|who is|what is)\b", low):
            tools.add("wikipedia")
        if re.search(r"\b(currency|usd|inr|eur|convert\s+money)\b", low):
            tools.add("currency")
        if re.search(r"\b(search|google|latest|news|find online)\b", low):
            tools.add("web_search")
            want_search = True
        if re.search(r"\b(document|according to|our policy|codebase|retrieve|rag)\b", low):
            use_rag = True
            tools.add("rag")
        if re.search(r"\b(run\s+tests|shell|bash|execute\s+code|python\s+script)\b", low):
            want_sandbox = True
            tools.add("bash")
            tools.add("code_execution")
        if re.search(r"\b(which tool|tool search|available tools)\b", low):
            want_search = True
            tools.add("tool_search")

        # Low confidence factual → allow search even without keyword
        if conf < 0.4 and arou > 0.3 and not tools:
            tools.add("web_search")
            want_search = True

        return tools, use_rag, want_search, want_sandbox


# Module-level singleton used by serve / agent
_default_bhutu: Optional[Bhutu] = None


def get_bhutu() -> Bhutu:
    global _default_bhutu
    if _default_bhutu is None:
        import os
        sandbox = (os.environ.get("NEUROFIELD_SANDBOX") or "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
        _default_bhutu = Bhutu(enable_sandbox=sandbox)
    return _default_bhutu


def bhutu_decide(prompt: str, **kwargs: Any) -> BhutuDecision:
    return get_bhutu().decide(prompt, **kwargs)
