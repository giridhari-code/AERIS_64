"""Megha tool adapter — defensive security assistant for orchestrator / agents."""

from __future__ import annotations

from typing import Any, Optional

from sentinel.megha import MeghaRequest, run_megha, list_capabilities


def megha_report(
    text: str = "",
    *,
    mode: str = "auto",
    target_type: str = "web",
    authorized: bool = False,
    scope: str = "",
    attestor: str = "",
    framework: str = "general",
) -> str:
    """Human-readable Megha output for prompt injection / tool notes."""
    if (mode or "auto").lower() in ("auto", ""):
        # Prefer policy routing
        from sentinel.megha.policy import evaluate_user_text

        pol = evaluate_user_text(text or "", authorized=authorized, scope=scope)
        if not pol.allow:
            return f"[Megha policy] BLOCKED: {pol.user_message or pol.reason}"
        mode = pol.megha_mode or "checklist"

    req = MeghaRequest(
        mode=mode,
        text=text or "",
        target_type=target_type,
        framework=framework,
        authorized=authorized,
        scope=scope,
        attestor=attestor,
    )
    r = run_megha(req)
    lines = [f"[Megha] mode={r.mode} ok={r.ok}"]
    if r.summary:
        lines.append(f"  summary: {r.summary}")
    if r.error:
        lines.append(f"  error: {r.error}")
    for it in (r.items or [])[:15]:
        if isinstance(it, dict):
            lines.append("  - " + "; ".join(f"{k}={v}" for k, v in it.items() if v is not None))
        else:
            lines.append(f"  - {it}")
    for w in (r.warnings or [])[:3]:
        lines.append(f"  note: {w}")
    return "\n".join(lines)


def megha_scan_dict(
    text: str = "",
    *,
    mode: str = "help",
    **kwargs: Any,
) -> dict[str, Any]:
    req = MeghaRequest(mode=mode, text=text, **{k: v for k, v in kwargs.items() if k in MeghaRequest.__dataclass_fields__})
    return run_megha(req).to_dict()


def megha_capabilities_text() -> str:
    caps = list_capabilities()
    lines = ["[Megha] capabilities:"]
    for c in caps:
        auth = " (needs auth)" if c.get("needs_auth") else ""
        lines.append(f"  - {c.get('id')}: {c.get('title')}{auth}")
    return "\n".join(lines)
