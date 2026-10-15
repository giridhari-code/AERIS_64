"""Megha policy monitor — block unauthorized / illegal cyber activity at the control layer.

Works with Metacognition as the *intent gate* before tools or model generation:
  - Defensive, educational, authorized assessment → allow (often route to Megha)
  - Unauthorized intrusion, malware, credential theft, exploit weaponization → refuse

This cannot stop a determined offline attacker; it enforces product policy on this API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Optional


# Offensive / clearly unauthorized patterns (English + common Hinglish cues)
_BLOCK_PATTERNS: list[tuple[str, str]] = [
    (r"\b(hack(ing)?\s+(into|someone|their|his|her|wifi|account|instagram|facebook|whatsapp|gmail))\b", "unauthorized_access"),
    (r"\b(break\s+into|gain\s+access\s+without|without\s+(permission|authori[sz]ation))\b", "unauthorized_access"),
    (r"\b(steal\s+(password|credential|cookie|token|otp|session)|dump\s+password)\b", "credential_theft"),
    (r"\b(brute\s*force\s+(login|password|ssh)|crack\s+(password|wifi|wpa))\b", "credential_attack"),
    (r"\b(ransomware|keylogger|rootkit|botnet|c2\s*server|reverse\s*shell\s+payload)\b", "malware"),
    (r"\b(write|build|generate|create)\s+(a\s+)?(virus|malware|trojan|worm)\b", "malware"),
    (r"\b(sql\s*injection\s+(on|against|into)\s+(their|his|production|live))\b", "offensive_exploit"),
    (r"\b(exploit\s+(cve-|zero[\s-]*day)|weaponize\s+exploit)\b", "offensive_exploit"),
    (r"\b(ddos|denial\s+of\s+service|stress\s+test\s+against\s+https?://)\b", "availability_attack"),
    (r"\b(phishing\s+(kit|page|mail)|clone\s+(bank|login)\s+page)\b", "social_engineering"),
    (r"\b(carding|cvv\s*shop|fullz|stolen\s+cards)\b", "fraud"),
    (r"\b(bypass\s+(paywall|license|activation)|piracy\s+crack)\b", "piracy"),
    # Hinglish / casual
    (r"\b(kisi\s+ka\s+(account|wifi|password)\s+(hack|tod|tor))\b", "unauthorized_access"),
    (r"\b(binas\s+permission|bina\s+permission|unauthorized\s+hack)\b", "unauthorized_access"),
]

# Defensive / legitimate educational cues → prefer Megha, do not block.
# Security-specific on purpose: generic words ("patch", "legal", "my server",
# "my app") used to hijack normal chat into Megha checklists.
_ALLOW_HINTS = re.compile(
    r"\b("
    r"owasp|hardening|checklist|secure\s+code|code\s+review|incident\s+response|"
    r"authorized\s+(pentest|assessment|test)|bug\s*bounty|responsible\s+disclosure|"
    r"defend|defensive|blue\s*team|soc\s+alert|cve\s+mitigation|"
    r"pen(?:etration)?\s*test|vulnerability\s+(scan|review|assessment)|threat\s+model|"
    r"security\s+(review|audit|checklist|testing|headers)|"
    r"secret\s+scan|log\s+triage|"
    r"iocs?|indicators?\s+of\s+compromise|defang|"
    r"compliance\s+(map|checklist|review)|gdpr|iso\s*27001|pci\s*dss"
    r")\b",
    re.I,
)

# Legit defensive work *mentions* attack keywords: detection rules, incident
# triage, protection/hardening. When these cues are present, route to Megha
# playbooks instead of refusing (Megha never outputs attack steps anyway).
_DEFENSIVE_CONTEXT = re.compile(
    r"\b("
    r"triage|incident|breach|postmortem|forensic[s]?|"
    r"detect(?:ion|or)?|yara|sigma|ids|ips|siem|monitor(?:ing)?|alert[s]?|"
    r"mitigat\w*|protect\w*|defen[cs]e|defensive|harden\w*|prevent\w*|"
    r"recover\w*|remediation|backup[s]?|response|redact\w*|quarantin\w*"
    r")\b",
    re.I,
)


@dataclass
class PolicyDecision:
    allow: bool
    action: str  # "allow" | "refuse" | "route_megha" | "require_authz"
    category: str = "ok"
    reason: str = ""
    user_message: str = ""
    megha_mode: Optional[str] = None
    metacognition: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _match_block(text: str) -> Optional[tuple[str, str]]:
    for pat, cat in _BLOCK_PATTERNS:
        if re.search(pat, text, flags=re.I):
            return cat, pat
    return None


def evaluate_user_text(text: str, *, authorized: bool = False, scope: str = "") -> PolicyDecision:
    """Metacognition-style control decision for one user utterance."""
    raw = (text or "").strip()
    if not raw:
        return PolicyDecision(allow=True, action="allow", reason="empty")

    blocked = _match_block(raw)
    if blocked:
        cat, _ = blocked
        # Even with authorized=true, malware/fraud/phishing kits stay refused
        hard = cat in {"malware", "fraud", "social_engineering", "piracy", "availability_attack"}
        # Defensive framing (detect / triage / protect / prevent): the user is
        # defending, not attacking → Megha playbooks, never a refusal.
        if _DEFENSIVE_CONTEXT.search(raw):
            low = raw.lower()
            mode = (
                "incident_triage"
                if re.search(r"triage|incident|breach|ransom|forensic|postmortem", low)
                else "checklist"
            )
            return PolicyDecision(
                allow=True,
                action="route_megha",
                category=cat,
                reason="defensive_context_route_megha",
                megha_mode=mode,
                metacognition={"policy_route": "megha", "defensive_override": True},
            )
        if hard or not authorized:
            msg = (
                "Megha policy + metacognition control: ye request **block** hai. "
                "Unauthorized access, malware, credential theft, ya dusre systems pe attack "
                "is product me allowed nahi. "
                "Apne systems ke liye: authorized assessment plan, checklist, secure code review, "
                "ya incident triage use karo (mode=assess_plan ke saath likhi permission)."
            )
            return PolicyDecision(
                allow=False,
                action="refuse",
                category=cat,
                reason=f"blocked_pattern:{cat}",
                user_message=msg,
                metacognition={
                    "policy_refuse": True,
                    "gate_bias": -1.0,
                    "want_replay": False,
                    "k": 1,
                    "freeze_writes": True,
                },
            )
        # authorized + non-hard offensive language → still force Megha plan, no free-form attack help
        return PolicyDecision(
            allow=True,
            action="route_megha",
            category=cat,
            reason="authorized_but_sensitive_route_megha",
            megha_mode="assess_plan",
            user_message="",
            metacognition={"policy_route": "megha", "k": 1, "gate_bias": -0.5},
        )

    if _ALLOW_HINTS.search(raw):
        # Security-positive → Megha can help
        mode = "checklist"
        low = raw.lower()
        if "code" in low or "review" in low:
            mode = "code_review"
        elif "incident" in low or "breach" in low or "triage" in low:
            mode = "incident_triage"
        elif "compliance" in low or "gdpr" in low or "iso" in low:
            mode = "compliance_map"
        elif "threat" in low or "stride" in low:
            mode = "threat_model"
        elif "defang" in low or "ioc" in low:
            mode = "ioc_extract"
        elif "secret" in low and "scan" in low:
            mode = "secret_scan"
        elif "pentest" in low or "assessment" in low or "scope" in low:
            mode = "assess_plan" if authorized else "checklist"
        return PolicyDecision(
            allow=True,
            action="route_megha",
            category="defensive",
            reason="defensive_intent",
            megha_mode=mode,
            metacognition={"policy_route": "megha", "prefer_tools": False},
        )

    return PolicyDecision(
        allow=True,
        action="allow",
        category="ok",
        reason="no_security_policy_hit",
        metacognition={"policy_route": "default"},
    )


def apply_meta_overrides(decision: PolicyDecision) -> dict[str, Any]:
    """Signals the serve path can apply (freeze memory writes, force refuse)."""
    return dict(decision.metacognition or {})
