"""Defensive playbooks and checklists — no exploit content."""

from __future__ import annotations

from typing import Any


CAPABILITIES = [
    {
        "id": "checklist",
        "title": "Security checklist",
        "desc": "Baseline hardening / review checklist for web, API, or host.",
        "needs_auth": False,
    },
    {
        "id": "code_review",
        "title": "Secure code review",
        "desc": "Heuristic review of provided source for common vulnerability classes.",
        "needs_auth": False,
    },
    {
        "id": "config_review",
        "title": "Config / header review",
        "desc": "Review config snippets or HTTP security headers.",
        "needs_auth": False,
    },
    {
        "id": "assess_plan",
        "title": "Authorized assessment plan",
        "desc": "Build a legal pentest/assessment plan for an attested scope.",
        "needs_auth": True,
    },
    {
        "id": "incident_triage",
        "title": "Incident triage",
        "desc": "Debug / triage steps from logs or incident description.",
        "needs_auth": False,
    },
    {
        "id": "compliance_map",
        "title": "Compliance process map",
        "desc": "High-level legal process steps (awareness only, not legal advice).",
        "needs_auth": False,
    },
    {
        "id": "debug_security",
        "title": "Security debugging",
        "desc": "Help debug auth, TLS, CORS, secrets handling issues from symptoms.",
        "needs_auth": False,
    },
]


def web_api_checklist() -> list[str]:
    return [
        "Enforce HTTPS/TLS 1.2+; redirect HTTP → HTTPS; HSTS where appropriate",
        "Authenticate every sensitive endpoint; prefer short-lived tokens + rotation",
        "Authorize by role/resource (never trust client-side checks alone)",
        "Validate and encode all inputs; parameterized queries only (no string-built SQL)",
        "Rate-limit auth and expensive APIs; lockout / backoff on brute force",
        "CSRF protection on cookie-based session state changes",
        "Secure cookies: HttpOnly, Secure, SameSite; CSP and frame protections",
        "Centralized error handling — no stack traces or secrets in client responses",
        "Dependency scanning and timely patching; pin versions in production",
        "Secrets in a vault/env — never in git; audit key access",
        "Logging of auth failures and admin actions without logging passwords/tokens",
        "Backup and recovery tested; least-privilege service accounts",
    ]


def host_checklist() -> list[str]:
    return [
        "OS and packages patched on a defined schedule",
        "Unnecessary services disabled; firewall default-deny",
        "SSH: key auth preferred, no root login, fail2ban or equivalent",
        "Disk encryption for sensitive hosts; secure wipe on decommission",
        "Central logging + time sync; integrity monitoring on critical binaries",
        "Separate admin and user accounts; MFA for remote admin",
    ]


def code_review_heuristics(source: str) -> list[dict[str, Any]]:
    """Lightweight static heuristics on provided text — not a full SAST engine."""
    findings: list[dict[str, Any]] = []
    lower = source.lower()
    rules = [
        ("sql_concat", ["execute(", "cursor.execute", "select * from", "\"select ", "'select "],
         "Possible dynamic SQL — prefer parameterized queries / ORM binds", "high"),
        ("command_inject", ["os.system(", "subprocess.call(", "shell=true", "popen("],
         "OS command execution — avoid shell=True; sanitize args; prefer allowlists", "high"),
        ("hardcoded_secret", ["api_key=", "apikey=", "password=", "secret=", "aws_secret", "private_key"],
         "Possible hardcoded credential — move to env/vault and rotate", "high"),
        ("insecure_tls", ["verify=false", "ssl._create_unverified", "insecure_skip_verify"],
         "TLS verification disabled — only for local debug, never production", "medium"),
        ("debug_left_on", ["debug=true", "app.debug", "flask_env=development"],
         "Debug mode indicators — disable in production builds", "medium"),
        ("eval_exec", ["eval(", "exec(", "pickle.loads("],
         "Dynamic code / unsafe deserialize — high RCE risk if input-controlled", "high"),
        ("path_traversal", ["../", "..\\\\", "send_file(", "open(user"],
         "Check path confinement (resolve + root prefix) for user-influenced paths", "medium"),
    ]
    for fid, needles, msg, sev in rules:
        hits = [n for n in needles if n in lower]
        if hits:
            findings.append({
                "id": fid,
                "severity": sev,
                "message": msg,
                "matched": hits[:5],
            })
    if not findings:
        findings.append({
            "id": "none_matched",
            "severity": "info",
            "message": "No heuristic patterns matched; still run full SAST/peer review.",
            "matched": [],
        })
    return findings


def config_header_review(text: str) -> list[dict[str, str]]:
    lower = text.lower()
    tips: list[dict[str, str]] = []
    checks = [
        ("strict-transport-security", "Consider HSTS for HTTPS sites"),
        ("content-security-policy", "CSP reduces XSS impact — start report-only then enforce"),
        ("x-content-type-options", "nosniff helps block MIME sniffing"),
        ("x-frame-options", "Or CSP frame-ancestors to mitigate clickjacking"),
        ("access-control-allow-origin: *", "Wildcard CORS with credentials is dangerous"),
    ]
    for needle, tip in checks:
        if needle in lower:
            tips.append({"check": needle, "status": "seen", "note": tip})
        else:
            tips.append({"check": needle, "status": "missing_or_not_in_snippet", "note": tip})
    return tips


def assessment_plan(scope: str, focus: str = "web") -> list[dict[str, str]]:
    """High-level authorized assessment phases — no exploit steps."""
    return [
        {"phase": "1. Scoping", "actions": f"Confirm written authorization; document in-scope assets for: {scope}; define out-of-scope and emergency contacts"},
        {"phase": "2. Recon (passive)", "actions": "Public records, DNS, certificate transparency, architecture docs — no invasive probes without go-ahead"},
        {"phase": "3. Mapping", "actions": "Enumerate allowed services/APIs from docs and agreed methods only"},
        {"phase": "4. Controlled testing", "actions": f"Run agreed test cases for {focus}: authz, input validation, session, crypto, config — stop on production impact"},
        {"phase": "5. Evidence", "actions": "Screenshot/log proof with timestamps; no exfiltration of real PII beyond need-to-prove"},
        {"phase": "6. Report", "actions": "Severity, repro steps, impact, remediation, retest plan; responsible disclosure timeline"},
        {"phase": "7. Remediation support", "actions": "Verify fixes; update runbooks and monitoring"},
    ]


def incident_triage_steps(symptom: str) -> list[str]:
    return [
        f"Capture symptom window: {symptom[:200]}",
        "Preserve logs (auth, app, reverse proxy, WAF) — do not wipe evidence",
        "Identify blast radius: accounts, hosts, data classes touched",
        "Contain: rotate compromised credentials, isolate host/token, rate-limit abuse",
        "Eradicate root cause (patch, revoke keys, fix bug) after containment",
        "Recover services from known-good state; monitor for recurrence",
        "Post-incident: timeline, lessons, control improvements",
        "If legal/regulatory trigger (breach of personal data), notify counsel/DPO per policy — Megha is not legal advice",
    ]


def compliance_map(framework: str = "general") -> list[dict[str, str]]:
    """Process awareness only — not legal advice."""
    base = [
        {"step": "Inventory", "detail": "Systems, data types, processors, owners"},
        {"step": "Risk assessment", "detail": "Threats, likelihood, impact, residual risk"},
        {"step": "Controls", "detail": "Technical + organizational controls mapped to risks"},
        {"step": "Policies", "detail": "Access, retention, incident response, vendor management"},
        {"step": "Evidence", "detail": "Logs, tickets, training records for audits"},
        {"step": "Incident process", "detail": "Detect → contain → notify (if required) → remediate → review"},
        {"step": "Review cycle", "detail": "Periodic control testing and management sign-off"},
    ]
    note = {
        "general": "Generic security governance flow",
        "gdpr": "Add lawful basis, DSR handling, DPA/vendor clauses, breach notification clocks — consult counsel",
        "iso27001": "Align with ISMS: Annex A controls, SoA, internal audit, management review",
        "pci": "Cardholder data environment scope, SAQ/RoC path, key management — QSA for formal validation",
    }.get(framework.lower(), "Generic security governance flow")
    return [{"step": "context", "detail": note}, *base]


def debug_security(symptom: str) -> list[str]:
    s = symptom.lower()
    tips = ["Reproduce with minimal case; capture exact status codes and headers"]
    if "cors" in s:
        tips += [
            "Check Access-Control-Allow-Origin vs actual Origin (no * with credentials)",
            "Ensure preflight OPTIONS handled; Allow-Headers/Methods match request",
        ]
    if "tls" in s or "ssl" in s or "certificate" in s:
        tips += [
            "Verify chain completeness and clock skew",
            "Disable obsolete protocols/ciphers; test with SSLLabs or openssl s_client (your systems only)",
        ]
    if "auth" in s or "401" in s or "403" in s or "jwt" in s:
        tips += [
            "Confirm token expiry, audience, issuer; skew tolerance",
            "Separate authentication failure (401) from authorization (403)",
            "Never log raw tokens; redact Authorization headers",
        ]
    if "secret" in s or "key" in s:
        tips += [
            "Rotate exposed secrets immediately",
            "Scan git history; use secret scanner in CI",
        ]
    tips.append("Add regression test and monitoring once fixed")
    return tips
