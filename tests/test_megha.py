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
