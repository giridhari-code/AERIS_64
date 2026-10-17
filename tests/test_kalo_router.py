from sentinel.kalo.router import route_intent
from sentinel.kalo.agent import KaloBot, _dataset_action


def test_routes():
    assert route_intent("buddy status").target == "buddy"
    assert route_intent("/buddy on").target == "buddy"
    assert route_intent("--buddy off").target == "buddy"
    assert route_intent("dataset progress").target == "dataset"
    assert route_intent("export training jsonl").target == "dataset"
    assert route_intent("token budget").target == "dataset"
    assert route_intent("weather in Delhi").target == "tools"
    assert route_intent("megha security review").target == "megha"
    assert route_intent("hello there").target == "chat"


def test_dataset_action():
    assert _dataset_action("please balance the dataset") == "balance"
    assert _dataset_action("export training jsonl") == "export_train"
    assert _dataset_action("show progress") == "progress"


def test_kalo_run_dataset_without_model():
    turn = KaloBot(generate=None).run("dataset progress")
    assert turn.route == "dataset"
    assert turn.events
    assert turn.events[0].name == "dataset_builder"
