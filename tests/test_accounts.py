"""Local accounts: signup/login store + /v1/auth/* API + required-mode gate."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from sentinel.config import ModelConfig, SafetyConfig
from sentinel.model import SENTINEL
from sentinel.serving.accounts import AccountError, AccountStore
from sentinel.serving.server import create_app
from sentinel.tokenizer.char import CharTokenizer

from sentinel.checkpoint import save_checkpoint

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :.?!\n"


# ---------------------------------------------------------------- store unit


@pytest.fixture()
def store(tmp_path):
    return AccountStore(tmp_path / "auth" / "accounts.db")


def test_store_signup_login_verify_logout(store):
    user, token = store.signup("Asha", "Asha@Example.com", "correct-horse")
    assert user["email"] == "asha@example.com"  # normalized
    assert store.verify(token)["id"] == user["id"]

    # login is case-insensitive on email; old token stays valid (multi-device)
    user2, token2 = store.login("asha@EXAMPLE.com", "correct-horse")
    assert user2["id"] == user["id"] and token2 != token
    assert store.verify(token) and store.verify(token2)

    # logout revokes exactly one session
    assert store.logout(token2) is True
    assert store.verify(token2) is None
    assert store.verify(token)["email"] == "asha@example.com"
    assert store.logout("not-a-token") is False


def test_store_rejects_duplicates_and_bad_input(store):
    store.signup("Asha", "asha@example.com", "correct-horse")
    with pytest.raises(AccountError) as dup:
        store.signup("Other", "ASHA@example.com", "another-pass")
    assert dup.value.status == 409

    with pytest.raises(AccountError) as weak:
        store.signup("Bo", "bo@example.com", "short")
    assert weak.value.status == 422

    with pytest.raises(AccountError) as badmail:
        store.signup("Bo", "not-an-email", "long-enough-pass")
    assert badmail.value.status == 422

    with pytest.raises(AccountError) as wrong:
        store.login("asha@example.com", "wrong-password")
    assert wrong.value.status == 401

    with pytest.raises(AccountError) as nomail:
        store.login("ghost@example.com", "whatever-pass")
    assert nomail.value.status == 401


def test_store_token_expiry(store, monkeypatch):
    import time

    monkeypatch.setenv("NEUROFIELD_AUTH_TTL_DAYS", str(1e-5))  # ≈ 0.86 s
    _, token = store.signup("Asha", "asha@example.com", "correct-horse")
    assert store.verify(token) is not None
    time.sleep(1.1)
    assert store.verify(token) is None


def test_store_hashes_are_salted(store):
    user, _ = store.signup("Asha", "asha@example.com", "correct-horse")
    import sqlite3

    conn = sqlite3.connect(store.path)
    row = conn.execute("SELECT pw_salt, pw_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
    conn.close()
    assert len(row[0]) == 16 and len(row[1]) == 32
    assert b"correct-horse" not in row[1]


# ------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt") / "tiny"
    tok = CharTokenizer.from_text(CHARS)
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=3, top_k=2, k_max=2,
        vocab_size=tok.vocab_size, dendrite_window=2, max_seq_len=256,
    )
    torch.manual_seed(0)
    model = SENTINEL(cfg, SafetyConfig(enable_audit=False))
    out = save_checkpoint(root, model, config={}, meta={"model_id": "unit-test"})
    (out / "tokenizer.json").write_text(json.dumps(tok.to_vocab_dict()), encoding="utf-8")
    return out


@pytest.fixture()
def client(tmp_path, monkeypatch, ckpt_dir):
    monkeypatch.delenv("NEUROFIELD_API_KEYS", raising=False)
    monkeypatch.delenv("NEUROFIELD_AUTH", raising=False)
    monkeypatch.delenv("NEUROFIELD_AUTH_REQUIRED", raising=False)
    monkeypatch.setenv("NEUROFIELD_AUTH_DB", str(tmp_path / "accounts.db"))
    return TestClient(create_app(checkpoint=str(ckpt_dir), device="cpu"))


def test_auth_status_shape(client):
    s = client.get("/v1/auth/status").json()
    assert s["accounts"] is True
    assert s["guest_ok"] is True and s["required"] is False
    assert s["users"] == 0


def test_auth_api_full_flow(client):
    r = client.post("/v1/auth/signup", json={"name": "Asha", "email": "asha@example.com", "password": "correct-horse"})
    assert r.status_code == 201
    body = r.json()
    token = body["token"]
    assert body["user"]["name"] == "Asha"
    assert "correct-horse" not in json.dumps(body)  # no plaintext password echo

    assert client.get("/v1/auth/status").json()["users"] == 1

    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["user"]["email"] == "asha@example.com"

    assert client.get("/v1/auth/me").status_code == 401
    assert client.get("/v1/auth/me", headers={"Authorization": "Bearer nope"}).status_code == 401

    dup = client.post("/v1/auth/signup", json={"name": "Asha", "email": "ASHA@example.com", "password": "correct-horse"})
    assert dup.status_code == 409

    bad = client.post("/v1/auth/login", json={"email": "asha@example.com", "password": "wrong"})
    assert bad.status_code == 401

    login = client.post("/v1/auth/login", json={"email": "asha@example.com", "password": "correct-horse"})
    assert login.status_code == 200
    token2 = login.json()["token"]

    assert client.post("/v1/auth/logout", headers={"Authorization": f"Bearer {token2}"}).status_code == 200
    assert client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token2}"}).status_code == 401
    assert client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_auth_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROFIELD_AUTH", "0")
    monkeypatch.setenv("NEUROFIELD_AUTH_DB", str(tmp_path / "accounts.db"))
    app = create_app(checkpoint=str(ckpt_dir), device="cpu")
    c = TestClient(app)
    s = c.get("/v1/auth/status").json()
    assert s["accounts"] is False and s["guest_ok"] is True
    assert c.post("/v1/auth/signup", json={"name": "A", "email": "a@b.co", "password": "long-enough"}).status_code == 404


def test_required_mode_blocks_chat_without_login(client, monkeypatch):
    # (client fixture already sets NEUROFIELD_AUTH_DB for this app instance)
    monkeypatch.setenv("NEUROFIELD_AUTH_REQUIRED", "1")
    assert client.get("/v1/auth/status").json()["guest_ok"] is False

    no_token = client.post("/v1/chat/stream", json={"prompt": "hi", "max_new_tokens": 2, "temperature": 0})
    assert no_token.status_code == 401

    token = client.post(
        "/v1/auth/signup", json={"name": "Asha", "email": "asha@example.com", "password": "correct-horse"}
    ).json()["token"]
    ok = client.post(
        "/v1/chat/stream",
        json={"prompt": "hi", "max_new_tokens": 2, "temperature": 0},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok.status_code == 200
    assert "data:" in ok.text


def test_required_mode_api_key_still_works(ckpt_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROFIELD_AUTH_REQUIRED", "1")
    monkeypatch.setenv("NEUROFIELD_API_KEYS", "machine-key-1")
    monkeypatch.setenv("NEUROFIELD_AUTH_DB", str(tmp_path / "accounts.db"))
    app = create_app(checkpoint=str(ckpt_dir), device="cpu")
    c = TestClient(app)
    ok = c.post(
        "/v1/chat/stream",
        json={"prompt": "hi", "max_new_tokens": 2, "temperature": 0},
        headers={"X-API-Key": "machine-key-1"},
    )
    assert ok.status_code == 200
    assert c.post("/v1/chat/stream", json={"prompt": "hi", "max_new_tokens": 2}).status_code == 401
