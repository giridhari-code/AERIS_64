"""Buddy v2 commands: mimic (joker), mood sensing, stats, who/link, about, animations.

Pure unit tests — no model/app needed (the API surface for commands is covered
in tests/test_buddy_egg.py).
"""

from sentinel.buddy.eggs import EggStore, required_steps

from sentinel.buddy import handle_buddy_command, is_buddy_command, resolve_buddy_reply, sense_mood


def test_mimic_echoes_given_text():
    out = handle_buddy_command("/buddy mimic Hello World", "probe")
    assert out and out.startswith("BUDDY_ANIM:excited")
    msg, meta = resolve_buddy_reply(out)
    assert "hElLo wOrLd" in msg and "nakal" in msg.lower()
    assert meta["buddy_anim"] == "excited" and meta["buddy"] is True


def test_mimic_reads_last_message_and_empty_hint():
    out = resolve_buddy_reply(handle_buddy_command("/buddy mimic", "probe", last_text="chai peene chalo"))[0]
    assert "chai" in out.lower() and "Hehe" in out
    hint = resolve_buddy_reply(handle_buddy_command("/buddy mimic", "probe", last_text=""))[0]
    assert "pehle kuch bolo" in hint.lower()


def test_joker_alias():
    msg, meta = resolve_buddy_reply(handle_buddy_command("--buddy joker test", "probe"))
    assert "Joker mode" in msg and meta["buddy_anim"] == "excited"


def test_mood_sensing_per_emotion():
    msg, meta = resolve_buddy_reply(handle_buddy_command("/buddy mood", "probe", last_text="aaj bahut khush hoon!"))
    assert "happy" in msg and meta["buddy_anim"] == "happy"
    sad = resolve_buddy_reply(handle_buddy_command("buddy mood", "probe", last_text="main udaas hoon"))[0]
    assert "sad" in sad
    angry = resolve_buddy_reply(handle_buddy_command("/buddy mood", "probe", last_text="gussa aa raha hai"))[0]
    assert "angry" in angry
    neutral = resolve_buddy_reply(handle_buddy_command("/buddy mood", "probe", last_text=""))[0]
    assert "neutral" in neutral


def test_sense_mood_units():
    assert sense_mood("thanks, badhiya!") == "happy"
    assert sense_mood("thak gaya hoon") == "tired"
    assert sense_mood("bahut pyaar") == "loving"
    assert sense_mood("random plain text") == "neutral"
    assert sense_mood("") == "neutral"


def test_stats_who_about():
    stats = handle_buddy_command("/buddy stats", "probe")
    assert stats and "curiosity" in stats and "█" in stats and "Personality" in stats
    who = handle_buddy_command("/buddy who", "probe")
    assert who and "probe" in who and "AERIS-64" in who and "Same login" in who
    about = handle_buddy_command("--buddy about", "probe")
    assert about and "AERIS-64" in about and "chat" in about


def test_animation_commands():
    for cmd, anim in (
        ("/buddy dance", "excited"),
        ("/buddy party", "excited"),
        ("/buddy wave", "wave"),
        ("/buddy happy", "happy"),
    ):
        msg, meta = resolve_buddy_reply(handle_buddy_command(cmd, "probe"))
        assert meta["buddy_anim"] == anim and msg


def test_help_and_detection():
    help_txt = handle_buddy_command("/buddy help", "probe")
    assert help_txt
    for bit in ("mimic", "mood", "stats", "about", "--buddy on"):
        assert bit in help_txt
    assert is_buddy_command("buddy mood")
    assert is_buddy_command("--buddy on")
    assert is_buddy_command("/buddy stats")
    assert not is_buddy_command("hello buddy")
    assert not is_buddy_command("buddy can you fix my code")


def test_resolve_tokens():
    assert resolve_buddy_reply("BUDDY_ON")[1] == {"buddy": True, "buddy_ui": "on"}
    assert resolve_buddy_reply("BUDDY_OFF")[1]["buddy_ui"] == "off"
    msg, meta = resolve_buddy_reply("BUDDY_EGG_UI")
    assert meta["buddy_egg"] is True and "float" in msg.lower()
    msg2, meta2 = resolve_buddy_reply("plain reply")
    assert msg2 == "plain reply" and meta2 == {"buddy": True}


def test_adopt_store_pins_buddy_and_goal(tmp_path):
    store = EggStore(tmp_path / "eggs.json")
    req = store.snapshot("web-dev-1")["required"]
    store.add_message_steps("web-dev-1", 2)

    res = store.adopt("web-dev-1", "u-acc-1")
    assert res["adopted"] is True and res["buddy_key"] == "web-dev-1"
    assert store.buddy_key("u-acc-1") == "web-dev-1"
    assert store.buddy_key("web-dev-1") is None  # moved, not copied

    snap = store.snapshot("u-acc-1")
    assert snap["messages"] == 2
    assert snap["required"] == req == required_steps("web-dev-1")  # goal preserved

    # a started account egg (messages > 0) keeps its own companion
    store.add_message_steps("u-acc-2", 1)
    store.add_message_steps("web-dev-2", 1)
    res2 = store.adopt("web-dev-2", "u-acc-2")
    assert res2["adopted"] is False and str(res2["reason"]).startswith("account")
    assert store.buddy_key("u-acc-2") is None

    # an untouched account egg (float auto-created it on login) is replaced —
    # this is the login race where the float loads before the adopt lands
    store.snapshot("u-acc-4")
    store.add_message_steps("web-dev-4", 2)
    res4 = store.adopt("web-dev-4", "u-acc-4")
    assert res4["adopted"] is True
    assert store.snapshot("u-acc-4")["messages"] == 2
    assert store.buddy_key("u-acc-4") == "web-dev-4"

    # nothing to adopt → no-op
    res3 = store.adopt("web-nope", "u-acc-3")
    assert res3["adopted"] is False and res3["reason"] == "no guest egg to adopt"

    # same identity → no-op
    assert store.adopt("u-acc-1", "u-acc-1")["adopted"] is False
