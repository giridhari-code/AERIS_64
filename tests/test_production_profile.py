import os
from sentinel.serving.production import load_profile, is_production, security_headers


def test_dev_allows_demo(monkeypatch):
    monkeypatch.delenv("NEUROFIELD_ENV", raising=False)
    monkeypatch.delenv("PRODUCTION", raising=False)
    monkeypatch.delenv("NEUROFIELD_PRODUCTION", raising=False)
    p = load_profile()
    assert not p.production
    assert p.allow_demo_checkpoint


def test_prod_strict(monkeypatch):
    monkeypatch.setenv("NEUROFIELD_ENV", "production")
    p = load_profile()
    assert p.production
    assert p.require_api_keys
    assert not p.allow_demo_checkpoint
    issues = p.issues(has_api_keys=False, checkpoint_set=False)
    assert any("API_KEYS" in x for x in issues)
    assert any("CHECKPOINT" in x for x in issues)


def test_security_headers():
    h = security_headers()
    assert h["X-Content-Type-Options"] == "nosniff"
