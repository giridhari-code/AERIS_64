"""0z code company agent tests."""

from sentinel.zeroz.policy import check_user_message, check_tool_name
from sentinel.zeroz.planner import build_plan
from sentinel.zeroz.runtime import run_zeroz
from sentinel.zeroz.config import ZeroZConfig
from sentinel.zeroz.toolbus import ToolBus


def test_policy_blocks_code():
    d = check_user_message("write a python function to sort a list")
    assert d.allow is False


def test_policy_allows_research():
    d = check_user_message("What is the weather in Mumbai?")
    assert d.allow is True


def test_tool_block_bash():
    d = check_tool_name("bash")
    assert d.allow is False


def test_tool_allow_weather():
    d = check_tool_name("weather")
    assert d.allow is True


def test_plan_weather():
    p = build_plan("weather in Delhi")
    assert any(s.tool == "weather" for s in p.steps)


def test_run_refuse():
    r = run_zeroz("implement a rust http server", cfg=ZeroZConfig(use_model_generate=False))
    assert r.stopped_reason == "policy_refuse"


def test_run_time():
    r = run_zeroz("what time is it?", cfg=ZeroZConfig(use_model_generate=False))
    assert r.stopped_reason in ("answered", "no_tools", "facts_only")
    assert len(r.answer) > 5


def test_toolbus_schemas():
    bus = ToolBus()
    names = {s["name"] for s in bus.schemas()}
    assert "weather" in names
    assert "bash" not in names
    assert "code_execution" not in names
