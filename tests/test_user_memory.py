"""Per-user persistent memory: lossless transcripts + persisted slow ring (fast memory is never saved)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sentinel.checkpoint import save_checkpoint  # noqa: E402
from sentinel.config import ModelConfig, SafetyConfig  # noqa: E402
from sentinel.model import SENTINEL  # noqa: E402
from sentinel.serving import chat_history  # noqa: E402
from sentinel.serving.server import create_app  # noqa: E402
from sentinel.serving.user_memory import UserMemoryStore, cap_slow_norm  # noqa: E402
from sentinel.tokenizer.char import CharTokenizer  # noqa: E402

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :.?!\n"
SID = "window-aaaa1111"


# ----------------------------------------------------------------- the store itself
def test_transcript_is_lossless_ordered_and_unicode_safe(tmp_path):
    st = UserMemoryStore(tmp_path)
    long = "x" * 5000 + " \u0928\u092e\u0938\u094d\u0924\u0947 \u2728"
    for i in range(30):
        st.append_turn("k", "alice", SID, "user" if i % 2 == 0 else "assistant", f"{i}: {long}")
    t = st.transcript("k", "alice", SID, limit=1000)
    assert [x["text"][:3] for x in t][:3] == ["0: ", "1: ", "2: "] and len(t) == 30
    assert all(x["text"].endswith("\u2728") for x in t)  # nothing truncated (the in-RAM window caps at 500 chars)


def test_rotation_never_deletes_and_torn_line_is_tolerated(tmp_path):
    st = UserMemoryStore(tmp_path, max_chat_mb=0.01)  # ~10 KB files
    for i in range(40):
        st.append_turn("k", "u", SID, "user", f"turn {i} " + "y" * 900)
    assert len(st.transcript("k", "u", SID, limit=1000)) == 40  # spread over several files, all kept
    files = list((tmp_path).rglob("*.jsonl"))
    assert len(files) > 1
    newest = max(files, key=lambda p: p.stat().st_mtime)
    with open(newest, "a", encoding="utf-8") as f:
        f.write('{"ts": 1, "role": "user", "te')  # crash mid-write
    assert len(st.transcript("k", "u", SID, limit=1000)) == 40


def test_owner_and_user_isolation_and_delete(tmp_path):
    st = UserMemoryStore(tmp_path)
    st.append_turn("key-one", "alice", SID, "user", "secret")
    assert st.transcript("key-two", "alice", SID) == []   # other API key
    assert st.transcript("key-one", "bob", SID) == []     # other user
    assert st.delete_user("key-one", "alice") is True
    assert st.transcript("key-one", "alice", SID) == [] and st.delete_user("key-one", "alice") is False


def test_slow_state_roundtrip_is_exact_and_corrupt_file_is_ignored(tmp_path):
    st = UserMemoryStore(tmp_path)
    slow = {"slow_slots": torch.randn(4, 8, 8), "slow_filled": torch.tensor([3.0, 1, 0, 0]),
            "slow_ptr": torch.tensor([2], dtype=torch.int64)}
    st.save_slow("k", "u", slow)
    back = st.load_slow("k", "u")
    assert all(torch.equal(back[k], slow[k]) for k in slow)
    (st._user_dir("k", "u") / "slow.safetensors").write_bytes(b"garbage")
    assert st.load_slow("k", "u") is None


def test_norm_cap():
    slow = {"slow_slots": torch.stack([torch.ones(4, 4) * 10, torch.ones(4, 4) * 0.1])}
    capped, n, biggest = cap_slow_norm(slow, 5.0)
    assert n == 1 and biggest == pytest.approx(40.0)
    assert capped["slow_slots"][0].norm() == pytest.approx(5.0, rel=1e-4)
    assert torch.equal(capped["slow_slots"][1], slow["slow_slots"][1])


# ----------------------------------------------------------------- end-to-end through the API
@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt") / "tiny"
    tok = CharTokenizer.from_text(CHARS)
    cfg = ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=3, top_k=2, k_max=2,
                      vocab_size=tok.vocab_size, dendrite_window=2, max_seq_len=256)
    torch.manual_seed(0)
    out = save_checkpoint(root, SENTINEL(cfg, SafetyConfig(enable_audit=False)), config={}, meta={"model_id": "mem-test"})
    (out / "tokenizer.json").write_text(json.dumps(tok.to_vocab_dict()), encoding="utf-8")
    return out


def _app(ckpt_dir, monkeypatch, mem_dir, keys="key-one,key-two"):
    if keys:
        monkeypatch.setenv("NEUROFIELD_API_KEYS", keys)
    else:
        monkeypatch.delenv("NEUROFIELD_API_KEYS", raising=False)
    monkeypatch.setenv("NEUROFIELD_MEMORY_DIR", str(mem_dir))
    return TestClient(create_app(checkpoint=str(ckpt_dir), device="cpu"))


def _chat(c, text, key="key-one", **kw):
    body = {"prompt": text, "session_id": SID, "user_id": "alice", "temperature": 0.0,
            "max_new_tokens": 6, "auto_length": False, **kw}
    r = c.post("/v1/chat", json=body, headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    return r.json()


def test_chat_persists_transcript_and_slow_memory_and_survives_restart(ckpt_dir, tmp_path, monkeypatch):
    chat_history.reset_history()
    c = _app(ckpt_dir, monkeypatch, tmp_path)
    H = {"X-API-Key": "key-one"}
    _chat(c, "hello there how are you")
    info = c.get("/v1/memory?user_id=alice", headers=H).json()
    assert info["chat_bytes"] > 0 and info["slow_memory_saved"] is True
    t1 = c.get(f"/v1/memory/transcript?session_id={SID}&user_id=alice", headers=H).json()["turns"]
    assert [x["role"] for x in t1] == ["user", "assistant"] and t1[0]["text"] == "hello there how are you"

    st = UserMemoryStore(tmp_path)
    s1 = st.load_slow("key-one", "alice")
    writes1 = float(s1["slow_filled"].sum())
    assert 0 < writes1 < 20                      # replayed history did not flood the ring

    # --- "restart": new app object, in-process chat window gone, same disk -------------------
    chat_history.reset_history()
    c2 = _app(ckpt_dir, monkeypatch, tmp_path)
    _chat(c2, "what did i just say to you")
    assert chat_history.history_len(SID) >= 2     # window rebuilt from the transcript, not lost
    t2 = c2.get(f"/v1/memory/transcript?session_id={SID}&user_id=alice", headers=H).json()["turns"]
    assert len(t2) == 4 and t2[0]["text"] == "hello there how are you"   # old turns still there, verbatim
    s2 = st.load_slow("key-one", "alice")
    assert float(s2["slow_filled"].sum()) > writes1                      # ring grew: context accumulates
    audit = (st._user_dir("key-one", "alice") / "audit.jsonl").read_text().strip().splitlines()
    assert len(audit) == 2 and json.loads(audit[0])["event"] == "slow_memory_saved"


def test_fast_memory_is_not_persisted_and_other_users_are_isolated(ckpt_dir, tmp_path, monkeypatch):
    chat_history.reset_history()
    c = _app(ckpt_dir, monkeypatch, tmp_path)
    _chat(c, "remember the code word is banana")
    saved = UserMemoryStore(tmp_path).load_slow("key-one", "alice")
    assert set(saved) == {"slow_slots", "slow_filled", "slow_ptr"}      # no M / h / x_hist on disk
    other_key = c.get("/v1/memory?user_id=alice", headers={"X-API-Key": "key-two"}).json()
    assert other_key["chat_bytes"] == 0 and other_key["slow_memory_saved"] is False
    assert c.get("/v1/memory?user_id=alice").status_code == 401


def test_remember_false_and_delete(ckpt_dir, tmp_path, monkeypatch):
    chat_history.reset_history()
    c = _app(ckpt_dir, monkeypatch, tmp_path)
    H = {"X-API-Key": "key-one"}
    _chat(c, "private chat", remember=False)
    assert c.get("/v1/memory?user_id=alice", headers=H).json()["chat_bytes"] == 0
    _chat(c, "keep this one")
    assert c.get("/v1/memory?user_id=alice", headers=H).json()["chat_bytes"] > 0
    assert c.delete("/v1/memory?user_id=alice", headers=H).json()["status"] == "deleted"
    after = c.get("/v1/memory?user_id=alice", headers=H).json()
    assert after["chat_bytes"] == 0 and after["slow_memory_saved"] is False
    assert chat_history.history_len(SID) == 0


def test_memory_stays_off_without_api_keys(ckpt_dir, tmp_path, monkeypatch):
    c = _app(ckpt_dir, monkeypatch, tmp_path, keys=None)
    r = c.post("/v1/chat", json={"prompt": "hi there", "user_id": "alice", "max_new_tokens": 4})
    assert r.status_code == 200
    assert not list(tmp_path.rglob("*.jsonl"))            # nothing written to disk
    assert c.get("/v1/memory").status_code == 503


def test_short_chat_turns_still_fill_the_users_slow_memory(ckpt_dir, tmp_path, monkeypatch):
    """A turn shorter than segment_len crosses no ring boundary; the turn-end consolidation must still store it."""
    chat_history.reset_history()
    c = _app(ckpt_dir, monkeypatch, tmp_path)
    _chat(c, "hi", max_new_tokens=3)
    st = UserMemoryStore(tmp_path)
    assert float(st.load_slow("key-one", "alice")["slow_filled"].sum()) >= 1
    first = float(st.load_slow("key-one", "alice")["slow_filled"].sum())
    _chat(c, "ok", max_new_tokens=3)
    assert float(st.load_slow("key-one", "alice")["slow_filled"].sum()) > first   # each turn adds to the ring
