"""Defensive playbooks and checklists — no exploit content."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
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
    {
        "id": "threat_model",
        "title": "Threat model (STRIDE)",
        "desc": "STRIDE-style threat prompts + controls for a described system. Fully local.",
        "needs_auth": False,
    },
    {
        "id": "secret_scan",
        "title": "Secret scan",
        "desc": "Scan provided text for keys/tokens (redacted output). Fully local.",
        "needs_auth": False,
    },
    {
        "id": "log_triage",
        "title": "Log triage",
        "desc": "Local log summary: levels, HTTP errors, repeated patterns, bad signals.",
        "needs_auth": False,
    },
    {
        "id": "report",
        "title": "Findings report",
        "desc": "Turn findings into a clean markdown report (offline).",
        "needs_auth": False,
    },
    {
        "id": "ioc_extract",
        "title": "IOC extraction",
        "desc": "Pull URLs/domains/IPs/hashes/CVEs from text, defanged. Fully local.",
        "needs_auth": False,
    },
    {
        "id": "sarif",
        "title": "SARIF export",
        "desc": "Findings → SARIF 2.1.0 JSON for CI/code-scanning UIs. Offline.",
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


def cloud_checklist() -> list[str]:
    return [
        "Identity: least-privilege IAM, no long-lived root/admin keys; MFA on root, console, and CI",
        "Storage: block public access by default; review bucket/container policies; encryption at rest on",
        "Network: default-deny security groups/NACLs; no 0.0.0.0/0 on admin ports; data in private subnets",
        "Logging: CloudTrail/Activity/audit logs routed to a locked-down account; alert on IAM changes",
        "Secrets: managed secret store + rotation; no secrets in user-data, images, or repo",
        "Metadata service hardening (IMDSv2-only) to blunt SSRF credential theft",
        "Key management: KMS/HSM with rotation and separation of duties",
        "Backups + tested restore; versioning/object-lock for ransomware-worthy data",
        "Guardrails: org policies / service control policies stop dangerous configs org-wide",
        "Cost/anomaly alerts as an early signal of cryptomining or stolen credentials",
    ]


def container_checklist() -> list[str]:
    return [
        "Images from trusted registries only; pin by digest; scan in CI (Trivy/Grype), block on criticals",
        "Run as non-root; read-only root filesystem; drop capabilities; no privileged / hostPath",
        "No secrets in images or env — mount from a secret store and rotate",
        "Resource requests/limits set (CPU, memory, PIDs) to contain noisy neighbors and forks",
        "Network policies default-deny pod-to-pod; restrict egress (metadata endpoints especially)",
        "K8s RBAC least privilege; no cluster-admin for app service accounts; audit logging on",
        "API server not public; admission control (OPA/Kyverno) blocks risky specs",
        "etcd encrypted + access restricted; control-plane upgrade cadence tracked",
        "Verify image provenance (cosign/signatures) where supply chain matters",
        "Runtime detections (Falco-class) for shell spawns, miners, unexpected outbound",
    ]


def mobile_checklist() -> list[str]:
    return [
        "No API keys/credentials baked into the bundle — assume the app can be unpacked",
        "Tokens in Keystore/Keychain, not SharedPreferences/UserDefaults/plain files",
        "TLS everywhere; certificate pinning if the threat model includes hostile networks",
        "Validate deep links/intents; don't export Android components you don't need",
        "WebView: no JS bridges to sensitive APIs; restrict file access and origins",
        "Root/jailbreak signals only as risk input — never the sole gate for secrets",
        "No sensitive data in logs, screenshots, or clipboard; mask PII in the UI",
        "Minimum OS versions + patch cadence; track MASVS/OWASP MASTG coverage",
        "Server-side authorization — never trust client-side checks alone",
        "Release builds: obfuscation (R8/ProGuard), no debug flags, signing keys in hardware/HSM",
    ]


def detect_stack(source: str) -> list[str]:
    """Best-effort stack hints from provided text (local, no network)."""
    s = (source or "").lower()
    stacks: list[str] = []
    if "<?php" in s or "$_post" in s or "$_get" in s:
        stacks.append("PHP")
    if "fastapi" in s:
        stacks.append("Python/FastAPI")
    elif "django" in s:
        stacks.append("Python/Django")
    elif "flask" in s or "@app.route" in s:
        stacks.append("Python/Flask")
    if "express" in s or "app.use(" in s:
        stacks.append("Node/Express")
    if "package main" in s and "func " in s:
        stacks.append("Go")
    if "public class " in s and "import java" in s:
        stacks.append("Java")
    if "server {" in s and "location /" in s:
        stacks.append("Nginx")
    if "apiversion:" in s and "kind:" in s:
        stacks.append("Kubernetes manifest")
    if "services:" in s and "image:" in s:
        stacks.append("Docker Compose")
    return stacks


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
        ("ssrf", ["requests.get(", "requests.post(", "httpx.get(", "urlopen(", "axios.get("],
         "Outbound request — if the URL is user-influenced: allowlist hosts, block metadata endpoints (169.254.169.254)", "high"),
        ("xxe", ["etree.parse(", "xml.etree", "lxml.etree", "minidom.parse(", "resolve_entities", "xml.sax"],
         "XML parsing — disable DTD/external entities; prefer defusedxml", "high"),
        ("jwt_weakness", ["jwt.decode(", "algorithm='none'", 'algorithm="none"', "verify_signature=false"],
         "JWT handling — pin allowed algorithms, verify exp/aud/iss, reject alg=none", "high"),
        ("unsafe_deserialize", ["yaml.load(", "marshal.loads(", "dill.loads(", "jsonpickle.decode"],
         "Unsafe deserialization — use safe loaders (yaml.safe_load) and never untrusted data", "high"),
        ("upload_risk", ["request.files[", "file.filename", "multipart/form-data"],
         "File upload — validate extension+content, randomize names, store outside web root, scan", "medium"),
        ("race_toctou", ["os.path.exists(", "if exists(", "shutil.move("],
         "Possible TOCTOU / race — check and act atomically (open with O_EXCL, temp files)", "low"),
        ("weak_random", ["random.random(", "random.randint(", "random.choice("],
         "random module is predictable — use secrets/os.urandom for tokens and IDs", "low"),
        ("cors_wildcard", ["access-control-allow-origin: *", 'allow_origins=["*"]', "allow_origins=['*']", "origin: *"],
         "Wildcard CORS — use explicit origins; never combine * with credentials", "medium"),
        ("ssti", ["render_template_string(", "jinja2.template(", "eval(template", "twig(", "freemarker"],
         "Template injection risk — never render user input as a template", "high"),
        ("open_redirect", ["redirect(request.args", "redirect(request.get", "redirect(req.query", "res.redirect(req."],
         "Open redirect — allowlist relative paths or known hosts only", "medium"),
        ("csrf_disabled", ["csrf_exempt", "csrf disabled", "csrf=false"],
         "CSRF protection disabled/exempt — re-enable or justify narrowly", "medium"),
        ("mass_assignment", ["**request.json", "**req.body", "update(**", "superset(**"],
         "Mass assignment — whitelist fields explicitly (avoid **request body into model)", "medium"),
        ("insecure_cookie", ["samesite=none", 'samesite="none"', "httponly=false", "secure=false"],
         "Cookie flags weakened — HttpOnly + Secure + SameSite=Lax/Strict", "medium"),
        ("weak_crypto", ["md5(", "sha1(", "des.new(", "rc4", "ecb mode", "mode_ecb", "aes-128-ecb", "aes-256-ecb"],
         "Weak crypto — use SHA-256+/AES-GCM; avoid MD5/SHA1/DES/ECB", "high"),
        ("zip_slip", ["extractall(", "tar.extract(", "zipfile.extract("],
         "Archive extraction — validate member paths (no ../ or absolute) before extracting", "high"),
        ("ldap_injection", ["ldap.search(", "ldap_search(", "search_s("],
         "LDAP query built from input — escape filter metacharacters", "medium"),
        ("nosql_injection", ["$where", "$ne:", "$gt:"],
         "NoSQL operator from input — cast/validate types; never pass raw objects into queries", "medium"),
        ("graphql_introspection", ["introspection: true", "__schema", "graphiql=true"],
         "GraphQL introspection/explorer — disable in production", "low"),
        ("log_injection", ['logger.info(f"', "log.info(req", "logger.info(request", "console.log(req."],
         "User data straight into logs — sanitize newlines/control chars; never log secrets", "low"),
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
    # ReDoS heuristic: nested quantifier shapes (e.g. (a+)+ , (.*)* )
    redos = re.findall(r"\([^()\n]*[+*][^()\n]*\)[+*]", source)
    if redos:
        findings.append({
            "id": "redos_risk",
            "severity": "medium",
            "message": "Nested quantifier — possible ReDoS if user input reaches this regex; test worst-case input",
            "matched": redos[:3],
        })
    stacks = detect_stack(source)
    if stacks:
        findings.append({
            "id": "stack_detected",
            "severity": "info",
            "message": "Detected stack(s): " + ", ".join(stacks) + " — apply stack-specific hardening",
            "matched": stacks,
        })
    if not findings:
        findings.append({
            "id": "none_matched",
            "severity": "info",
            "message": "No heuristic patterns matched; still run full SAST/peer review.",
            "matched": [],
        })
    return findings


# ---------- Secret scan (fully local, offline) ----------
_SECRET_PATTERNS: list[tuple[str, "re.Pattern[str]", str]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "high"),
    ("aws_secret_key", re.compile(r"(?i)aws[_\- ]?secret[_\- ]?(?:access[_\- ]?)?key\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{40}[\"']?"), "high"),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "high"),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), "high"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "high"),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "high"),
    ("stripe_secret", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{10,}\b"), "high"),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b"), "high"),
    ("mailgun_key", re.compile(r"\bkey-[a-f0-9]{32}\b"), "high"),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"), "high"),
    ("pypi_token", re.compile(r"\bpypi-[A-Za-z0-9_\-]{50,}\b"), "high"),
    ("digitalocean_token", re.compile(r"\bdop_v1_[a-f0-9]{64}\b"), "high"),
    ("twilio_sid", re.compile(r"\bAC[a-f0-9]{32}\b"), "medium"),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{33}\b"), "medium"),
    ("azure_conn_string", re.compile(r"(?i)AccountKey=[A-Za-z0-9+/=]{32,}"), "high"),
    ("basic_auth_url", re.compile(r"\bhttps?://[^\s/:@]+:[^\s/@]{4,}@"), "medium"),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "high"),
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}\b"), "medium"),
    ("generic_secret_assign", re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"), "medium"),
]


def _redact(value: str) -> str:
    if len(value) <= 10:
        return "*" * len(value)
    keep = 4
    return value[:keep] + "*" * max(4, min(20, len(value) - keep * 2)) + value[-keep:]


def _shannon_entropy(s: str) -> float:
    """Bits/char Shannon entropy — flags generic high-entropy strings locally."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def secret_scan(text: str) -> list[dict[str, Any]]:
    """Scan provided text for common credential shapes. Local only; output is redacted."""
    findings: list[dict[str, Any]] = []
    for ln, line in enumerate((text or "").splitlines(), 1):
        for fid, pat, sev in _SECRET_PATTERNS:
            m = pat.search(line)
            if m:
                findings.append({
                    "id": fid,
                    "severity": sev,
                    "line": ln,
                    "snippet": _redact(m.group(0)),
                    "fix": "Revoke/rotate this credential, purge it from git history, move to a secret store",
                })
    # Generic high-entropy strings that did not match a known shape
    seen: set[str] = set()
    for ln, line in enumerate((text or "").splitlines(), 1):
        for tok in re.findall(r"[A-Za-z0-9+/=_\-]{24,}", line):
            if tok in seen or len(findings) >= 40:
                continue
            seen.add(tok)
            if _shannon_entropy(tok) >= 3.6:
                findings.append({
                    "id": "high_entropy_string",
                    "severity": "medium",
                    "line": ln,
                    "snippet": _redact(tok),
                    "fix": "High-entropy string — if real, rotate it and move to a secret store",
                })
    if not findings:
        findings.append({
            "id": "no_secret_found",
            "severity": "info",
            "line": 0,
            "snippet": "",
            "fix": "No common credential shapes matched in the provided text (scan git history too)",
        })
    return findings


def threat_model(description: str, focus: str = "web") -> list[dict[str, Any]]:
    """STRIDE-style threat prompts + controls for a described system. Local, no lookups."""
    d = (description or "").lower()
    stride = [
        ("Spoofing", ["Can an attacker impersonate a user or service? (credential stuffing, weak MFA, session fixation)"],
         ["MFA on sensitive flows", "Strong session management + rotation", "Verify service-to-service identity (mTLS/tokens)"]),
        ("Tampering", ["Can inputs or stored data be modified?", "Is the CI/CD supply chain trusted?"],
         ["Server-side validation", "Integrity in transit/at rest (TLS, signatures)", "Signed artifacts + pinned dependencies"]),
        ("Repudiation", ["Could someone deny performing an action?"],
         ["Audit logs for auth + admin actions", "Tamper-evident log storage", "Correlation IDs across services"]),
        ("Information disclosure", ["What leaks via errors, logs, or over-broad APIs?"],
         ["Data classification + minimization", "Least privilege", "No secrets/stack traces in responses or logs"]),
        ("Denial of service", ["What happens under floods or expensive queries?"],
         ["Rate limits + quotas", "Timeouts and backpressure", "Autoscaling / circuit breakers"]),
        ("Elevation of privilege", ["Can a normal user reach admin actions?"],
         ["Server-side authz per resource (no client trust)", "Least-privilege service accounts", "Patch management"]),
    ]
    items = [
        {"category": cat, "threat_prompts": prompts, "controls": controls}
        for cat, prompts, controls in stride
    ]
    extra: list[str] = []
    if any(w in d for w in ("api", "endpoint", "rest", "graphql")):
        extra.append("API: object-level authorization (IDOR), per-key rate limits, schema validation")
    if any(w in d for w in ("mobile", "android", "ios", "app store")):
        extra.append("Mobile: bundle secrets, pinning, secure storage, tampered/hostile client")
    if any(w in d for w in ("cloud", "aws", "azure", "gcp", "s3", "bucket", "iam")):
        extra.append("Cloud: IAM breadth, storage exposure, metadata endpoint abuse, audit trails")
    if any(w in d for w in ("payment", "card", "pci", "wallet", "upi")):
        extra.append("Payments: PCI scope, tokenization, idempotency, fraud signals")
    items.append({
        "category": "Trust boundaries",
        "threat_prompts": [
            "Where does data cross a trust zone (user → app → DB → cloud → third party)?",
            "Is identity checked at every boundary, not just at the edge?",
        ],
        "controls": [
            "Authenticate + authorize at each boundary",
            "Encrypt in transit across zones",
            "Log cross-zone access; least privilege between zones",
        ],
    })
    if extra:
        items.append({"category": "Asset-specific", "threat_prompts": extra, "controls": []})
    if any(w in d for w in ("privacy", "pii", "personal data", "gdpr", "dpdp", "health", "medical", "user data", "kyc")):
        items.append({
            "category": "Privacy (LINDDUN-lite)",
            "threat_prompts": [
                "Linkability: can records be correlated to identify a person?",
                "Identifiability: is identifying data stored beyond need?",
                "Data disclosure: where could personal data leak (logs, backups, exports)?",
                "Unawareness: do users know what is collected and why?",
                "Non-compliance: are consent and purpose limits honored?",
            ],
            "controls": [
                "Data minimization + purpose limitation",
                "Pseudonymization / tokenization",
                "Retention limits + secure deletion",
                "Consent + notice flows; DSR handling",
            ],
        })
    return items


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
    cfg_checks = [
        ("autoindex on", "Nginx directory listing enabled — disable autoindex"),
        ("server_tokens on", "Nginx version disclosure — set server_tokens off"),
        ("ssl_protocols tlsv1 ", "Old TLS enabled — allow only TLSv1.2+"),
        ("privileged: true", "Container privileged mode — drop it; add only needed capabilities"),
        ("hostnetwork: true", "hostNetwork exposes the node network — prefer ClusterIP services"),
        ("hostpath:", "hostPath mounts the node filesystem — avoid or tightly scope"),
        ("allowprivilegeescalation: true", "allowPrivilegeEscalation should be false"),
        ("readonlyrootfilesystem: false", "Consider readOnlyRootFilesystem: true"),
        ("runasuser: 0", "Running as root — set a non-root runAsUser"),
        (":latest", "Pin image tags/digests instead of :latest"),
    ]
    for needle, note in cfg_checks:
        if needle in lower:
            tips.append({"check": needle, "status": "seen", "note": note})
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


# ---------- Incident runbooks (defensive, offline) ----------
_INCIDENT_EXTRAS: list[tuple[tuple[str, ...], list[str]]] = [
    (("phishing", "phish", "suspicious mail", "email link"), [
        "Preserve the original email (headers + attachments) before deleting anything",
        "Report to the mail provider; block sender/domain/URL centrally",
        "Check who clicked; force password reset + session revocation for them",
        "Review mailbox forwarding rules and OAuth app grants on affected accounts",
    ]),
    (("webshell", "shell upload", "backdoor", "cmd="), [
        "Find recently modified files in web roots; compare with known-good hashes",
        "Search access logs for POSTs to upload paths and unusual user agents",
        "Rotate app secrets the web process could read; rebuild host if uncertain",
    ]),
    (("bec", "wire transfer", "invoice fraud", "payment fraud"), [
        "Call the bank/payment provider now — recovery windows are short",
        "Preserve email threads; verify identity claims out-of-band (known phone number)",
        "Add dual-approval controls for payment detail changes",
    ]),
    (("supply chain", "dependency", "compromised package", "build system"), [
        "Pin/audit affected package versions; diff lockfiles against known-good",
        "Check build artifacts for unexpected changes; rebuild from clean runners",
        "Rotate credentials reachable by CI/CD; review pipeline permissions",
    ]),
    (("insider", "ex-employee", "terminated employee", "data theft"), [
        "Preserve evidence with chain-of-custody; involve HR/legal before interviews",
        "Review access logs, badge data, and mass export/download activity",
        "Revoke sessions and keys; rotate shared credentials the person knew",
    ]),
]


def incident_triage_steps(symptom: str) -> list[str]:
    steps = [
        f"Capture symptom window: {symptom[:200]}",
        "Preserve logs (auth, app, reverse proxy, WAF) — do not wipe evidence",
        "Identify blast radius: accounts, hosts, data classes touched",
        "Contain: rotate compromised credentials, isolate host/token, rate-limit abuse",
        "Eradicate root cause (patch, revoke keys, fix bug) after containment",
        "Recover services from known-good state; monitor for recurrence",
        "Post-incident: timeline, lessons, control improvements",
        "If legal/regulatory trigger (breach of personal data), notify counsel/DPO per policy — Megha is not legal advice",
    ]
    s = (symptom or "").lower()
    for keys, extra_steps in _INCIDENT_EXTRAS:
        if any(k in s for k in keys):
            steps.extend(extra_steps)
    return steps

# ---------- Log triage (fully local, offline) ----------
_LOG_LEVELS: list[tuple[str, str]] = [
    ("critical", r"\b(?:CRITICAL|FATAL|EMERG)\b"),
    ("error", r"\b(?:ERROR|ERR)\b"),
    ("warning", r"\bWARN(?:ING)?\b"),
    ("info", r"\bINFO\b"),
    ("debug", r"\bDEBUG\b"),
]

_LOG_SIGNALS: list[tuple[str, str, str, str]] = [
    ("auth_failures", r"failed password|authentication failure|invalid user|login failed|401",
     "Check lockout/backoff; confirm no credential stuffing wave", "T1110 Brute Force"),
    ("priv_esc_signals", r"sudo:|su: |not in sudoers|permission denied",
     "Review who/what escalated; separate service and admin accounts", "T1548 Abuse Elevation Control"),
    ("injection_attempts", r"union\s+select|'\s*or\s*'1'='1|<script|\.\./\.\./",
     "Web attack probes — verify WAF/validation; no sensitive error output", "T1190 Exploit Public-Facing App"),
    ("webshell_activity", r"cmd=|shell\.php|wget http|curl http|base64 -d|chmod \+x /tmp",
     "Possible webshell / payload staging — diff webroot, review upload paths", "T1505 Server Software Component"),
    ("resource_exhaustion", r"out of memory|oom-?kill|disk (?:is )?full|no space left|too many open files",
     "Capacity/abuse issue — raise limits deliberately, add alerts", "T1499 Endpoint DoS"),
    ("service_failure", r"segfault|core dumped|connection refused|connection reset|timed out|timeout",
     "Crash/dependency fault — capture stack, check deploys and upstreams", "T1489 Service Stop"),
    ("tls_cert_issues", r"certificate (?:expired|verify failed)|handshake failure|no such host",
     "Cert/trust issue — check expiry automation and clock skew", ""),
    ("crypto_miner_hint", r"stratum\+tcp|xmrig|minerd|kdevtmpfsi",
     "Possible compromised host mining crypto — isolate and investigate", "T1496 Resource Hijacking"),
]


def log_triage(text: str) -> list[dict[str, Any]]:
    """Summarize pasted log text locally: levels, HTTP errors, patterns, bad signals."""
    raw = text or ""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    levels = {
        name: len(re.findall(pat, raw))
        for name, pat in _LOG_LEVELS
    }
    c4 = len(re.findall(r"\b4\d\d\b", raw))
    c5 = len(re.findall(r"\b5\d\d\b", raw))

    signals: list[dict[str, Any]] = []
    for name, pat, hint, mitre in _LOG_SIGNALS:
        n = len(re.findall(pat, raw, flags=re.I))
        if n:
            entry: dict[str, Any] = {"signal": name, "count": n, "hint": hint}
            if mitre:
                entry["mitre"] = mitre
            signals.append(entry)

    norm: list[str] = []
    for ln in lines:
        s = re.sub(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", "HH:MM:SS", ln)
        s = re.sub(r"\b[0-9a-fA-F]{8,}\b", "HEX", s)
        s = re.sub(r"\b\d+\b", "N", s)
        norm.append(s.strip()[:160])
    top = Counter(norm).most_common(5)

    items: list[dict[str, Any]] = [
        {"summary": f"{len(lines)} non-empty line(s) analyzed"},
        {"levels": ", ".join(f"{k}:{v}" for k, v in levels.items() if v) or "no explicit levels"},
        {"http_status": f"4xx: {c4}, 5xx: {c5}"},
    ]
    if signals:
        items.extend(signals)
    else:
        items.append({"signals": "no known-bad markers"})
    stamps = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", raw)
    if stamps:
        items.append({"timeline": f"{stamps[0]} → {stamps[-1]} ({len(stamps)} timed event(s))"})
    items.append({"top_patterns": [
        {"pattern": p, "count": c} for p, c in top if c > 1
    ] or "no repeated patterns"})
    return items


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
    key = (framework or "general").lower().replace(" ", "").replace("-", "").replace("_", "")
    _pci = "Cardholder data environment scope, SAQ/RoC path, key management — QSA for formal validation"
    notes = {
        "general": "Generic security governance flow",
        "gdpr": "Add lawful basis, DSR handling, DPA/vendor clauses, breach notification clocks — consult counsel",
        "iso27001": "Align with ISMS: Annex A controls, SoA, internal audit, management review",
        "pci": _pci,
        "pcidss": _pci,
        "soc2": "Trust Services Criteria (security, availability, confidentiality); evidence over the observation window — Type II needs months of operating proof",
        "hipaa": "PHI scope + risk analysis; administrative/physical/technical safeguards; BAAs with vendors; breach notification duties — privacy officer + counsel",
        "nist": "NIST CSF 2.0 — Govern / Identify / Protect / Detect / Respond / Recover: map controls, score maturity, close gaps",
        "nistcsf": "NIST CSF 2.0 — Govern / Identify / Protect / Detect / Respond / Recover: map controls, score maturity, close gaps",
        "dpdp": "India DPDP Act 2023: notice + consent, purpose limitation, data-principal rights (access/correction/erasure), grievance officer, breach intimation to the Data Protection Board — counsel for legal",
    }
    note = notes.get(key, "Generic security governance flow")
    return [{"step": "context", "detail": note}, *base]


# ---------- Findings report + exports (fully local, offline) ----------
_REPORT_SEVERITIES = ("critical", "high", "medium", "low", "info")
_REPORT_WEIGHTS = {"critical": 10, "high": 5, "medium": 2, "low": 1, "info": 0}
_SEVERITY_TO_SARIF = {"critical": "error", "high": "error", "medium": "warning", "low": "note", "info": "note"}


def _parse_findings(findings_text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in (findings_text or "").splitlines():
        line = line.strip(" \t-*")
        if not line:
            continue
        sev, text = "info", line
        if ":" in line:
            head, rest = line.split(":", 1)
            head_l = head.strip().lower()
            if head_l in _REPORT_SEVERITIES or head_l == "crit":
                sev = "critical" if head_l == "crit" else head_l
                text = rest.strip() or head.strip()
        rows.append((sev, text))
    return rows


def _risk_band(score: int) -> str:
    if score <= 4:
        return "low"
    if score <= 12:
        return "moderate"
    if score <= 30:
        return "high"
    return "critical"


def build_report(findings_text: str, scope: str = "") -> str:
    """Turn line-separated findings ('high: SQLi at login') into a markdown report."""
    rows = _parse_findings(findings_text)
    counts = Counter(sev for sev, _ in rows)
    score = sum(_REPORT_WEIGHTS.get(sev, 0) for sev, _ in rows)
    md: list[str] = [
        "# Security Review — Findings Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} (local, offline)",
    ]
    if scope:
        md.append(f"**Scope:** {scope}")
    md += [
        "",
        "## Summary",
        f"- Total findings: {len(rows)}",
        f"- Risk score: {score} ({_risk_band(score)} exposure; critical=10 high=5 medium=2 low=1)",
    ]
    for sev in _REPORT_SEVERITIES:
        if counts.get(sev):
            md.append(f"- {sev.capitalize()}: {counts[sev]}")
    md += [
        "", "## Findings",
        "| # | Severity | Finding |",
        "|---|----------|---------|",
    ]
    for i, (sev, txt) in enumerate(rows, 1):
        md.append(f"| {i} | {sev} | {txt[:160].replace('|', '/')} |")
    md += [
        "", "## Next steps",
        "1. Fix critical/high findings first; add regression tests.",
        "2. Track medium/low with owners and dates.",
        "3. Re-test after fixes; keep evidence for audits.",
        "",
        "> Generated locally by Megha — defensive aid, not a penetration test or legal advice.",
    ]
    return "\n".join(md)


def build_findings_csv(findings_text: str) -> str:
    """Machine-readable remediation tracker (severity,status,owner,finding)."""
    rows = _parse_findings(findings_text)
    lines = ["severity,status,owner,finding"]
    for sev, txt in rows:
        safe = txt[:200].replace('"', '""')
        lines.append(f'"{sev}","open","","{safe}"')
    return "\n".join(lines)


def build_sarif(findings_text: str, scope: str = "") -> dict[str, Any]:
    """Minimal SARIF 2.1.0 export so CI/code-scanning UIs can consume findings."""
    rows = _parse_findings(findings_text)
    results = [
        {
            "ruleId": "megha.finding",
            "level": _SEVERITY_TO_SARIF.get(sev, "note"),
            "message": {"text": txt},
            "properties": {"severity": sev},
        }
        for sev, txt in rows
    ]
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "Megha",
                "rules": [{
                    "id": "megha.finding",
                    "name": "MeghaFinding",
                    "shortDescription": {"text": "Security review finding (local heuristic/playbook)"},
                }],
            }
        },
        "results": results,
    }
    if scope:
        run["properties"] = {"scope": scope}
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }


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


# ---------- IOC extraction (fully local, offline) ----------
_IOC_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("url", re.compile(r"\bhttps?://[^\s\"'<>\)\]]+", re.I)),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("sha256", re.compile(r"\b[a-fA-F0-9]{64}\b")),
    ("sha1", re.compile(r"\b[a-fA-F0-9]{40}\b")),
    ("md5", re.compile(r"\b[a-fA-F0-9]{32}\b")),
    ("cve", re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)),
    ("domain", re.compile(
        r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
        r"(?:com|net|org|io|in|co|dev|xyz|ru|cn|info|top|online|site|app|gov|edu|me|ai)\b",
        re.I,
    )),
]


def _defang(value: str) -> str:
    out = value.replace("https://", "hxxps://").replace("http://", "hxxp://")
    return out.replace(".", "[.]")


def _refang(value: str) -> str:
    """Normalize already-defanged input so extraction works either way."""
    return (
        value.replace("hxxps://", "https://")
        .replace("hxxp://", "http://")
        .replace("[.]", ".")
        .replace("(.)", ".")
    )


def _valid_ipv4(ip: str) -> bool:
    try:
        parts = [int(p) for p in ip.split(".")]
    except ValueError:
        return False
    return len(parts) == 4 and all(0 <= p <= 255 for p in parts)


def _is_private_ip(ip: str) -> bool:
    if not _valid_ipv4(ip):
        return False
    a, b = (int(p) for p in ip.split(".")[:2])
    return (
        a == 10 or a == 127 or a == 0
        or (a == 172 and 16 <= b <= 31)
        or (a == 192 and b == 168)
        or (a == 169 and b == 254)
    )


def ioc_extract(text: str) -> list[dict[str, Any]]:
    """Extract indicators (URL/domain/IP/hash/CVE/email) locally, defanged for safe sharing.

    Accepts both raw and already-defanged input (hxxp://, [.]).
    """
    raw = _refang(text or "")
    found: dict[str, list[str]] = {}
    for typ, pat in _IOC_PATTERNS:
        for m in pat.finditer(raw):
            v = m.group(0).rstrip(".,;)]")
            bucket = found.setdefault(typ, [])
            if v not in bucket:
                bucket.append(v)

    urls = found.get("url", [])
    items: list[dict[str, Any]] = []
    for typ in ("url", "domain", "email", "ipv4", "md5", "sha1", "sha256", "cve"):
        for v in found.get(typ, []):
            if typ == "domain" and any(v in u for u in urls):
                continue  # already covered inside a full URL
            if typ == "ipv4" and not _valid_ipv4(v):
                continue
            item: dict[str, Any] = {
                "type": typ,
                "value": _defang(v),
                "count": len(re.findall(re.escape(v), raw)),
            }
            if typ == "ipv4":
                item["private"] = _is_private_ip(v)
            items.append(item)
    if not items:
        items.append({"type": "none", "value": "no indicators matched", "count": 0})
    return items
