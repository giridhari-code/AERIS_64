"""Reliable agent loop smoke (no network required for calc path)."""

from sentinel.agent import AgentConfig, run_agent


def test_agent_calc_facts():
    r = run_agent("calculate 7*8", generate_fn=None, config=AgentConfig(max_steps=3))
    assert r.steps >= 1
    assert r.answer
    assert r.stopped_reason in {"facts_only", "answered", "no_model_no_facts", "generate_empty_fallback"}


def test_agent_empty():
    r = run_agent("  ", generate_fn=None)
    assert r.stopped_reason == "empty_input"


def test_agent_with_generate():
    r = run_agent(
        "hello",
        generate_fn=lambda p: "Hello from model",
        config=AgentConfig(max_steps=3),
    )
    assert "Hello" in r.answer
    assert r.stopped_reason == "answered"
