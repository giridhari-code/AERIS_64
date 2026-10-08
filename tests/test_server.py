"""End-to-end API tests on a tiny random-weight model (no external files)."""

import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from neurofield.checkpoint import save_checkpoint  # noqa: E402
from neurofield.config import ModelConfig, SafetyConfig  # noqa: E402
from neurofield.model import NeuroField  # noqa: E402
from neurofield.serving import chat_history  # noqa: E402
from neurofield.serving.server import create_app  # noqa: E402
from neurofield.tokenizer.char import CharTokenizer  # noqa: E402

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :.?!\n"


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt") / "tiny"
    tok = CharTokenizer.from_text(CHARS)
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=3, top_k=2, k_max=2,
        vocab_size=tok.vocab_size, dendrite_window=2, max_seq_len=256,
    )
    torch.manual_seed(0)
    model = NeuroField(cfg, SafetyConfig(enable_audit=False))
    out = save_checkpoint(
        root, model, config={}, meta={"model_id": "unit-test"},  # config deliberately EMPTY
    )
    (out / "tokenizer.json").write_text(json.dumps(tok.to_vocab_dict()), encoding="utf-8")
    return out


@pytest.fixture()
def client(ckpt_dir, monkeypatch):
    monkeypatch.delenv("NEUROFIELD_API_KEYS", raising=False)
    return TestClient(create_app(checkpoint=str(ckpt_dir), device="cpu"))


def test_empty_config_is_recovered_from_weights(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "unit-test" and body["params"] > 0


def test_greedy_chat_is_deterministic_and_fresh_state(client):
    a = client.post("/v1/chat", json={"prompt": "hello", "temperature": 0, "max_new_tokens": 12}).json()
    b = client.post("/v1/chat", json={"prompt": "hello", "temperature": 0, "max_new_tokens": 12}).json()
    assert a["text"] == b["text"]
    assert a["session_id"] != b["session_id"]


def test_reset_really_clears_history(client):
    sid = "session-aaaaaaaa"
    client.post("/v1/chat", json={"prompt": "hello", "session_id": sid, "temperature": 0, "max_new_tokens": 8})
    assert chat_history.history_len(sid) > 0
    assert client.post(f"/v1/reset?session_id={sid}").status_code == 200
    assert chat_history.history_len(sid) == 0


def test_bad_session_id_rejected(client):
    r = client.post("/v1/chat", json={"prompt": "hi there", "session_id": "../x"})
    assert r.status_code == 422


def test_stream_has_done_event_and_terminator(client):
    with client.stream("POST", "/v1/chat/stream",
                       json={"prompt": "hello", "temperature": 0, "max_new_tokens": 6}) as r:
        body = "".join(r.iter_text())
    assert '"event": "done"' in body and body.rstrip().endswith("data: [DONE]")


def test_completions_greedy_extends_input(client):
    ids = [1, 2, 3]
    j = client.post("/v1/completions", json={"input_ids": ids, "max_new_tokens": 5, "temperature": 0}).json()
    assert j["output_ids"][:3] == ids and len(j["output_ids"]) >= 4


def test_auth_header_only_and_session_owner(ckpt_dir, monkeypatch):
    monkeypatch.setenv("NEUROFIELD_API_KEYS", "key-one,key-two")
    c = TestClient(create_app(checkpoint=str(ckpt_dir), device="cpu"))
    body = {"prompt": "hello", "session_id": "owned-by-one-1", "temperature": 0, "max_new_tokens": 4}
    assert c.post("/v1/chat", json=body).status_code == 401
    assert c.post("/v1/chat?api_key=key-one", json=body).status_code == 401  # query param no longer works
    assert c.post("/v1/chat", json=body, headers={"X-API-Key": "key-one"}).status_code == 200
    assert c.post("/v1/chat", json=body, headers={"X-API-Key": "key-two"}).status_code == 403
    assert c.post("/v1/reset?session_id=owned-by-one-1", headers={"X-API-Key": "key-two"}).status_code == 403
