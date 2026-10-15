"""Megha engine — route intents to defensive playbooks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from sentinel.megha.authz import require_authorization, AuthorizationError
from sentinel.megha import playbooks
from sentinel.megha.policy import evaluate_user_text

# Modes whose `text` is content to review (code, config, logs, symptoms,
# descriptions) rather than an instruction — the keyword intent gate is
# skipped for these, and they all run fully local/offline.
_CONTENT_MODES = frozenset({
    "help", "capabilities", "list", "checklist", "compliance_map",
    "code_review", "config_review", "incident_triage", "debug_security",
    "threat_model", "secret_scan", "log_triage", "report", "sarif", "ioc_extract",
})


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
        # Control-layer pre-check on free text (metacognition policy monitor).
        # Only gate modes whose `text` is the *request*; content-review modes
        # receive code/config/logs/symptoms as material, and security keywords
        # there describe the subject, not intent (avoids false refusals).
        if text and mode not in _CONTENT_MODES:
            pol = evaluate_user_text(
                text, authorized=req.authorized, scope=req.scope or ""
            )
            if not pol.allow:
                return MeghaResult(
                    ok=False,
                    mode=mode,
                    error=pol.user_message or pol.reason,
                    warnings=warnings + [f"policy:{pol.category}"],
                )

        if mode in ("help", "capabilities", "list"):
            return MeghaResult(
                ok=True,
                mode="capabilities",
                summary="Available Megha modes",
                items=list_capabilities(),
                warnings=warnings,
            )

        if mode == "checklist":
            tt = (req.target_type or "web").strip().lower()
            if tt == "host":
                items = playbooks.host_checklist()
            elif tt in ("cloud", "aws", "azure", "gcp"):
                items = playbooks.cloud_checklist()
            elif tt in ("container", "docker", "k8s", "kubernetes"):
                items = playbooks.container_checklist()
            elif tt in ("mobile", "android", "ios"):
                items = playbooks.mobile_checklist()
            else:
                items = playbooks.web_api_checklist()
            return MeghaResult(
                ok=True,
                mode=mode,
                summary=f"Hardening checklist ({tt})",
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

        if mode == "threat_model":
            if not text:
                return MeghaResult(
                    ok=False, mode=mode,
                    error="text (system description) required",
                    warnings=warnings,
                )
            return MeghaResult(
                ok=True,
                mode=mode,
                summary=f"STRIDE threat model ({req.target_type}) — local, offline",
                items=playbooks.threat_model(text, focus=req.target_type),
                warnings=warnings,
            )

        if mode == "secret_scan":
            if not text:
                return MeghaResult(ok=False, mode=mode, error="text required", warnings=warnings)
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Secret scan of provided text (output redacted) — local, offline",
                items=playbooks.secret_scan(text),
                warnings=warnings + [
                    "Runs fully local — nothing leaves this process. Also scan git history (gitleaks/trufflehog)."
                ],
            )

        if mode == "log_triage":
            if not text:
                return MeghaResult(ok=False, mode=mode, error="text (log lines) required", warnings=warnings)
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Log triage — local, offline",
                items=playbooks.log_triage(text),
                warnings=warnings + ["Runs fully local — log text is not uploaded anywhere"],
            )

        if mode == "report":
            if not text:
                return MeghaResult(
                    ok=False, mode=mode,
                    error="text required (one finding per line, optional 'severity:' prefix)",
                    warnings=warnings,
                )
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Findings report (markdown + CSV tracker) — local, offline",
                items=[
                    {"markdown": playbooks.build_report(text, scope=req.scope or "")},
                    {"csv": playbooks.build_findings_csv(text)},
                ],
                warnings=warnings,
            )

        if mode == "sarif":
            if not text:
                return MeghaResult(
                    ok=False, mode=mode,
                    error="text required (one finding per line, optional 'severity:' prefix)",
                    warnings=warnings,
                )
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="SARIF 2.1.0 export — local, offline",
                items=[{"sarif": playbooks.build_sarif(text, scope=req.scope or "")}],
                warnings=warnings,
            )

        if mode == "ioc_extract":
            if not text:
                return MeghaResult(
                    ok=False, mode=mode,
                    error="text required (paste notes/logs/email headers)",
                    warnings=warnings,
                )
            return MeghaResult(
                ok=True,
                mode=mode,
                summary="Indicator extraction (defanged) — local, offline",
                items=playbooks.ioc_extract(text),
                warnings=warnings + ["Values defanged for safe sharing; extraction runs fully local"],
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
