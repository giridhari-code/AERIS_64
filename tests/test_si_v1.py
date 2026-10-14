
from sentinel.agent.si_v1 import build_plan, run_si_v1, SI_v1Config

def test_plan_calc():
    plan = build_plan("calculate 3*4")
    assert any(s["action"] == "calculator" for s in plan)

def test_run_calc():
    r = run_si_v1("calculate 3*4", config=SI_v1Config())
    assert "12" in r.answer

def test_empty():
    assert run_si_v1("  ").stopped_reason == "empty_input"
