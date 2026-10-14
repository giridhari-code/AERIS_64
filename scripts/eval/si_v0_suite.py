#!/usr/bin/env python3
"""SI_v0 task suite: tool reliability + optional model (no network fail = soft skip)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neurofield.agent import AgentConfig, run_agent


TASKS = [
    {"id": "calc_1", "q": "calculate 12*12", "expect_substr": ["144"]},
    {"id": "calc_2", "q": "what is 7+5", "expect_substr": ["12"]},
    {"id": "empty", "q": "   ", "expect_reason": "empty_input"},
    {"id": "time_or_facts", "q": "what time is it now", "expect_any_answer": True},
]


def main() -> int:
    cfg = AgentConfig(max_steps=4)
    results = []
    passed = 0
    for t in TASKS:
        r = run_agent(t["q"], generate_fn=None, config=cfg)
        ok = True
        if "expect_reason" in t:
            ok = r.stopped_reason == t["expect_reason"]
        if "expect_substr" in t:
            ok = ok and any(s in r.answer for s in t["expect_substr"])
        if t.get("expect_any_answer"):
            ok = ok and bool(r.answer.strip())
        passed += int(ok)
        results.append({
            "id": t["id"],
            "ok": ok,
            "stopped": r.stopped_reason,
            "answer_preview": r.answer[:120],
            "steps": r.steps,
        })
    report = {"passed": passed, "total": len(TASKS), "results": results}
    out = ROOT / "docs/audits/si_v0_suite_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")
    return 0 if passed == len(TASKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
