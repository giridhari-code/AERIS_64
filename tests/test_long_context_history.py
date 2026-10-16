"""Chat history long-context window caps."""
import os

def test_max_turns_default():
    os.environ.pop("NEUROFIELD_MAX_TURNS", None)
    os.environ.pop("MAX_TURNS", None)
    from sentinel.serving.chat_history import get_max_turns, reset_history, append_turn, history_len
    assert get_max_turns() == 24
    reset_history("t1")
    for i in range(60):
        append_turn("t1", "user", f"u{i}")
        append_turn("t1", "assistant", f"a{i}")
    # at most 24 pairs * 2
    assert history_len("t1") <= 48

def test_max_turns_env(monkeypatch):
    monkeypatch.setenv("NEUROFIELD_MAX_TURNS", "8")
    # re-import logic reads env each call
    from sentinel.serving import chat_history as ch
    assert ch.get_max_turns() == 8
