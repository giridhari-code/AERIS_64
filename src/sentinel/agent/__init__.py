"""Agent behaviour layers: SI_v0 (enrich) and SI_v1 (plan→tool→verify→answer)."""

from sentinel.agent.loop import AgentConfig, AgentResult, run_agent
from sentinel.agent.si_v1 import SI_v1Config, SI_v1Result, run_si_v1, build_plan

try:
    from sentinel.agent.generate import make_generate_fn
except Exception:  # pragma: no cover
    make_generate_fn = None  # type: ignore

__all__ = [
    "AgentConfig",
    "AgentResult",
    "run_agent",
    "SI_v1Config",
    "SI_v1Result",
    "run_si_v1",
    "build_plan",
    "make_generate_fn",
]
