"""Megha engine — route intents to defensive playbooks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from sentinel.megha.authz import require_authorization, AuthorizationError
from sentinel.megha import playbooks


@dataclass
class MeghaRequest:
    mode: str
    text: str = ""
    target_type: str = "web"  # web | host | api
    framework: str = "general"  # for compliance_map
    # assessment attestation
    authorized: bool = False
    scope: str = ""
    attestor: str = ""


@dataclass
class MeghaResult:
    ok: bool
    mode: str
    product: str = "Megha"
    summary: str = ""
    items: list[Any] = field(default_factory=list)
    authz: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def list_capabilities() -> list[dict]:
    return list(playbooks.CAPABILITIES)


def run_megha(req: MeghaRequest) -> MeghaResult:
    mode = (req.mode or "").strip().lower()
    text = (req.text or "").strip()
    warnings = [
        "Megha is a defensive assistant only. No unauthorized testing.",
        "Compliance maps are process awareness — not legal advice.",
    ]

    try:
        if mode in ("help", "capabilities", "list"):
            return MeghaResult(
                ok=True,
                mode="capabilities",
                summary="Available Megha modes",
                items=list_capabilities(),
                warnings=warnings,
            )

        if mode == "checklist":
            items = (
                playbooks.host_checklist()
                if req.target_type == "host"
                else playbooks.web_api_checklist()
            )
            return MeghaResult(
                ok=True,
                mode=mode,
                summary=f"Hardening checklist ({req.target_type})",
                items=[{"item": x} for x in items],
                warnings=warnings,
            )

        if mode == "code_review":
            if not text:
                return MeghaResult(ok=False, mode=mode, error="text (source snippet) required", warnings=warnings)
            findings = playbooks.code_review_heuristics(text)
            return MeghaResult(
                ok=True,
                mode=mode,
                summary=f"{len(findings)} heuristic finding(s)",
                items=findings,
                warnings=warnings + ["Heuristics only — use professional SAST for release gates"],
            )

        if mode == "config_review":
            if not text:
                return MeghaResult(ok=False, mode=mode, error="text (config/headers) required", warnings=warnings)
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Config / header review",
                items=playbooks.config_header_review(text),
                warnings=warnings,
            )

        if mode == "assess_plan":
            authz = require_authorization(
                authorized=req.authorized,
                scope=req.scope or text,
                attestor=req.attestor,
            )
            plan = playbooks.assessment_plan(authz["scope"], focus=req.target_type)
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Authorized assessment plan (no active scanning performed)",
                items=plan,
                authz=authz,
                warnings=warnings + ["Megha does not run network exploits or scans itself"],
            )

        if mode == "incident_triage":
            if not text:
                return MeghaResult(ok=False, mode=mode, error="text (symptom/logs summary) required", warnings=warnings)
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Incident triage playbook",
                items=[{"step": s} for s in playbooks.incident_triage_steps(text)],
                warnings=warnings,
            )

        if mode == "compliance_map":
            return MeghaResult(
                ok=True,
                mode=mode,
                summary=f"Compliance process map ({req.framework})",
                items=playbooks.compliance_map(req.framework),
                warnings=warnings,
            )

        if mode == "debug_security":
            if not text:
                return MeghaResult(ok=False, mode=mode, error="text (symptom) required", warnings=warnings)
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Security debugging hints",
                items=[{"tip": t} for t in playbooks.debug_security(text)],
                warnings=warnings,
            )

        return MeghaResult(
            ok=False,
            mode=mode or "unknown",
            error=f"Unknown mode '{mode}'. Use mode=help for capabilities.",
            items=list_capabilities(),
            warnings=warnings,
        )
    except AuthorizationError as e:
        return MeghaResult(ok=False, mode=mode, error=str(e), warnings=warnings)
