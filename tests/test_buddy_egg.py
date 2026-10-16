"""Buddy egg: Pokémon-style incubation store + /v1/buddy/step API.

Progress = 1 step per user message + 1 step per passive period
(NEUROFIELD_BUDDY_EGG_PERIOD, default 300 s). Requirement is deterministic
per user id (6–10). "Hatched" is derived, so the buddy itself never changes.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from sentinel.buddy.eggs import EggStore, egg_status_text, required_steps
from sentinel.config import ModelConfig, SafetyConfig
from sentinel.model import SENTINEL
from sentinel.serving.server import create_app
from sentinel.tokenizer.char import CharTokenizer

from sentinel.checkpoint import save_checkpoint

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :.?!\n"


# ---------------------------------------------------------------- store unit


@pytest.fixture()
def store(tmp_path):
    return EggStore(tmp_path / "buddy" / "eggs.json")


def test_store_creates_egg_and_is_deterministic(store):
    a = store.snapshot("alice")
    assert a["messages"] == 0 and a["progress"] == 0
    assert a["hatched"] is False
    assert 6 <= a["required"] <= 10
    assert a["required"] == required_steps("alice")
    assert a["frames"] and len(a["frames"]) >= 3
    assert a["crack_frames"] and len(a["crack_frames"]) >= 2

    # persisted on disk: a fresh store instance sees the same egg
    store2 = EggStore(store.path)
    b = store2.snapshot("alice")
    assert b["required"] == a["required"] and b["progress"] == 0
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert "alice" in data["users"]


def test_store_message_steps_and_hatch(store):
    req = store.snapshot("bob")["required"]
    last = None
    for i in range(req):
        last = store.add_message_steps("bob")
        assert last["hatched"] is (i + 1 >= req)
    assert last and last["progress"] == req and last["hatched"] is True

    # steps clamp at the requirement; egg stays hatched
    over = store.add_message_steps("bob", 5)
    assert over["progress"] == req and over["hatched"] is True
    assert "frames" not in over and over["next_step_in_s"] == 0

    # other users are unaffected
    other = store.snapshot("carol")
    assert other["progress"] == 0 and other["hatched"] is False


def test_store_passive_time_progress(store, monkeypatch):
    monkeypatch.setenv("NEUROFIELD_BUDDY_EGG_PERIOD", "100")
    t0 = 1000.0
    snap = store.snapshot("dina", now=t0)
    assert snap["progress"] == 0 and snap["next_step_in_s"] == 100

    t1 = store.snapshot("dina", now=t0 + 250)
    assert t1["progress"] == 2 and t1["next_step_in_s"] == 50

    # time alone can finish the egg ("kuch bhi karo ya na karo")
    t2 = store.snapshot("dina", now=t0 + t1["required"] * 100)
    assert t2["hatched"] is True and t2["progress"] == t1["required"]


def test_store_combo_messages_plus_time(store, monkeypatch):
    monkeypatch.setenv("NEUROFIELD_BUDDY_EGG_PERIOD", "10")
    t0 = 5000.0
    req = store.snapshot("eve", now=t0)["required"]
    snap = store.add_message_steps("eve", req - 3, now=t0)
    assert snap["progress"] == req - 3 and snap["hatched"] is False

    # 30 s later → 3 passive steps → hatched
    done = store.snapshot("eve", now=t0 + 30)
    assert done["hatched"] is True and done["progress"] == req


def test_period_env_and_default(monkeypatch):
    from sentinel.buddy import eggs

    monkeypatch.delenv("NEUROFIELD_BUDDY_EGG_PERIOD", raising=False)
    monkeypatch.delenv("BUDDY_EGG_PERIOD", raising=False)
    assert eggs.egg_period_s() == 300.0
    monkeypatch.setenv("NEUROFIELD_BUDDY_EGG_PERIOD", "not-a-number")
    assert eggs.egg_period_s() == 300.0
    monkeypatch.setenv("NEUROFIELD_BUDDY_EGG_PERIOD", "7.5")
    assert eggs.egg_period_s() == 7.5


def test_egg_status_text_and_buddy_command(store):
    from sentinel.buddy import handle_buddy_command

    snap = store.snapshot("frank")
    txt = egg_status_text(snap)
    assert "Buddy egg" in txt and f"/{snap['required']}" in txt and "```" in txt

    out = handle_buddy_command("/buddy", "frank", egg=snap)
    assert out and "Buddy egg" in out
    assert handle_buddy_command("/buddy on", "frank", egg=snap) == "BUDDY_ON"
    assert handle_buddy_command("/buddy off", "frank", egg=snap) == "BUDDY_OFF"

    done = snap
    for _ in range(snap["required"]):
        done = store.add_message_steps("frank")
    out2 = handle_buddy_command("/buddy", "frank", egg=done)
    assert out2 and "Buddy egg" not in out2  # revealed after hatching

    # no egg context (older callers) → legacy instant reveal
    legacy = handle_buddy_command("/buddy", "frank")
    assert legacy and "Buddy egg" not in legacy


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
    monkeypatch.delenv("NEUROFIELD_BUDDY_EGG_PERIOD", raising=False)
    monkeypatch.setenv("NEUROFIELD_BUDDY_DB", str(tmp_path / "buddy" / "eggs.json"))
    return TestClient(create_app(checkpoint=str(ckpt_dir), device="cpu"))


def test_buddy_api_reports_egg(client):
    r = client.get("/v1/buddy", params={"user_id": "api-egg-1"})
    assert r.status_code == 200
    j = r.json()
    assert j["buddy"]["name"]
    egg = j["egg"]
    assert egg["hatched"] is False and egg["progress"] == 0
    assert egg["frames"] and egg["crack_frames"]
    assert 6 <= egg["required"] <= 10
    assert egg["period_s"] == 300.0

    post = client.post("/v1/buddy", json={"command": "/buddy", "user_id": "api-egg-1"}).json()
    assert post["egg"]["hatched"] is False
    assert "Buddy egg" in post["text"]  # /buddy shows incubation, not the reveal


def test_buddy_step_api_hatches(client):
    uid = "api-egg-2"
    req = client.get("/v1/buddy", params={"user_id": uid}).json()["egg"]["required"]

    # one step short of the goal → still incubating (batched steps keep the
    # suite light on the shared rate limiter; per-step behavior is unit-tested)
    near = client.post("/v1/buddy/step", json={"user_id": uid, "n": req - 1}).json()
    assert near["hatched"] is False and near["egg"]["progress"] == req - 1

    last = client.post("/v1/buddy/step", json={"user_id": uid}).json()
    assert last["hatched"] is True
    assert last["egg"]["progress"] == req
    assert last["buddy"]["name"]
    assert last["egg"].get("frames") is None  # art trimmed once hatched

    post = client.post("/v1/buddy", json={"command": "/buddy", "user_id": uid}).json()
    assert post["egg"]["hatched"] is True
    assert "Buddy egg" not in (post["text"] or "")

    # further steps stay hatched and keep returning the companion
    again = client.post("/v1/buddy/step", json={"user_id": uid}).json()
    assert again["hatched"] is True and again["buddy"]["name"]


def test_buddy_step_clamps_and_isolates(client):
    a = client.post("/v1/buddy/step", json={"user_id": "api-egg-a", "n": 999}).json()
    assert a["egg"]["messages"] <= 25
    b = client.post("/v1/buddy/step", json={"user_id": "api-egg-b"}).json()
    assert b["egg"]["messages"] == 1


def test_chat_buddy_command_uses_user_id_and_egg(client):
    uid = "chat-egg-1"
    req = client.get("/v1/buddy", params={"user_id": uid}).json()["egg"]["required"]

    r = client.post("/v1/chat", json={"prompt": "/buddy", "user_id": uid})
    assert r.status_code == 200
    assert "Buddy egg" in r.json()["text"]

    client.post("/v1/buddy/step", json={"user_id": uid, "n": req})

    r2 = client.post("/v1/chat", json={"prompt": "/buddy", "user_id": uid})
    assert "Buddy egg" not in r2.json()["text"]


def test_ag_ui_buddy_command_egg_aware(client):
    uid = "agui-egg-1"
    payload = {
        "threadId": "t-egg-1",
        "runId": "r-egg-1",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "/buddy"}],
        "tools": [],
        "context": [],
        "forwardedProps": {"user_id": uid},
    }
    r = client.post("/ag-ui", json=payload)
    assert r.status_code == 200
    assert "Buddy egg" in r.text
