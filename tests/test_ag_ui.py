"""AG-UI protocol endpoint tests (spec 1.0 event stream) on a tiny random model."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
ag_ui = pytest.importorskip("ag_ui")  # noqa: F841 - optional dependency
from fastapi.testclient import TestClient  # noqa: E402

from sentinel.checkpoint import save_checkpoint  # noqa: E402
from sentinel.config import ModelConfig, SafetyConfig  # noqa: E402
from sentinel.model import SENTINEL  # noqa: E402
from sentinel.serving.server import create_app  # noqa: E402
from sentinel.tokenizer.char import CharTokenizer  # noqa: E402

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
    model = SENTINEL(cfg, SafetyConfig(enable_audit=False))
    out = save_checkpoint(root, model, config={}, meta={"model_id": "unit-test"})
    (out / "tokenizer.json").write_text(json.dumps(tok.to_vocab_dict()), encoding="utf-8")
    return out


@pytest.fixture()
def client(ckpt_dir, monkeypatch):
    monkeypatch.delenv("NEUROFIELD_API_KEYS", raising=False)
    return TestClient(create_app(checkpoint=str(ckpt_dir), device="cpu"))


def run_input(prompt: str, **props) -> dict:
    return {
        "threadId": "ag-ui-test-thread-1",
        "runId": "ag-ui-test-run-1",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": prompt}],
        "tools": [],
        "context": [],
        "forwardedProps": props or None,
    }


def sse_events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_ag_ui_streams_spec_events_and_sentinel_meta(client):
    r = client.post(
        "/ag-ui",
        json=run_input("hello", max_new_tokens=6, temperature=0, auto_length=False),
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "RUN_STARTED"
    assert "TEXT_MESSAGE_START" in kinds and "TEXT_MESSAGE_END" in kinds
    assert kinds[-1] == "RUN_FINISHED"
    assert any(e["type"] == "TEXT_MESSAGE_CONTENT" and e.get("delta") is not None for e in events)
    end = next(e for e in events if e["type"] == "TEXT_MESSAGE_END")
    sentinel = (end.get("metadata") or {}).get("sentinel") or {}
    assert sentinel.get("session_id") and sentinel.get("latency_ms") is not None


def test_ag_ui_alias_and_thread_continuity(client):
    r = client.post("/v1/ag-ui", json=run_input("hello again", max_new_tokens=4, temperature=0))
    assert r.status_code == 200
    events = sse_events(r.text)
    end = next(e for e in events if e["type"] == "TEXT_MESSAGE_END")
    assert (end.get("metadata") or {}).get("sentinel", {}).get("session_id") == "ag-ui-test-thread-1"


def test_ag_ui_buddy_command_short_circuits_model(client):
    r = client.post("/ag-ui", json=run_input("/buddy on"))
    assert r.status_code == 200
    events = sse_events(r.text)
    text = "".join(e.get("delta", "") for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "Buddy on" in text
    end = next(e for e in events if e["type"] == "TEXT_MESSAGE_END")
    assert (end.get("metadata") or {}).get("sentinel", {}).get("model") == "buddy"


def test_ag_ui_rejects_bad_body(client):
    assert client.post("/ag-ui", json={"threadId": "t", "runId": "r"}).status_code == 422
    assert client.post("/ag-ui", content=b"{not json").status_code == 400
