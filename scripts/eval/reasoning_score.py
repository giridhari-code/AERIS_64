#!/usr/bin/env python3
"""
Reasoning eval scores (system-matched).

Modes:
  - tool: SI_v1 agent (calculator etc.) — measures pipeline, not LM math
  - gold: compare expected Answer: field only (no model) — data integrity
  - model: optional --checkpoint generate (usually weak until SFT)

Reports accuracy % and writes docs/audits/reasoning_score_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.agent.si_v1 import run_si_v1, SI_v1Config

# Fixed held-out style items (not all identical to SFT file wording)
ITEMS = [
    {"id": "r01", "q": "calculate 3+4", "expect": "7", "mode": "tool"},
    {"id": "r02", "q": "calculate 12*5", "expect": "60", "mode": "tool"},
    {"id": "r03", "q": "calculate 100-37", "expect": "63", "mode": "tool"},
    {"id": "r04", "q": "calculate 8/2", "expect": "4", "mode": "tool"},
    {"id": "r05", "q": "calculate 7*8+1", "expect": "57", "mode": "tool"},
    {"id": "r06", "q": "calculate 2+2*3", "expect": "8", "mode": "tool"},
    {"id": "r07", "q": "what time is it", "expect_re": r"\d{4}-\d{2}-\d{2}|\d+:\d+", "mode": "tool"},
    {"id": "r08", "q": "calculate 15*3", "expect": "45", "mode": "tool"},
    {"id": "r09", "q": "calculate 9*9", "expect": "81", "mode": "tool"},
    {"id": "r10", "q": "calculate 50/5", "expect": "10", "mode": "tool"},
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _has_expect(answer: str, expect: str) -> bool:
    a = _norm(answer)
    e = _norm(expect)
    if e in a:
        return True
    # number forms: 12.0 vs 12
    if re.fullmatch(r"-?\d+(\.\d+)?", e):
        if e in a or f"{e}.0" in a:
            return True
        try:
            return str(float(e)) in a or str(int(float(e))) in a
        except ValueError:
            return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/audits/reasoning_score_report.json")
    args = ap.parse_args()

    rows = []
    ok_n = 0
    for it in ITEMS:
        q = it["q"]
        r = run_si_v1(q, config=SI_v1Config(max_steps=6))
        ans = r.answer
        passed = False
        if "expect" in it:
            passed = _has_expect(ans, it["expect"])
        elif "expect_re" in it:
            passed = bool(re.search(it["expect_re"], ans, re.I))
        ok_n += int(passed)
        rows.append({
            "id": it["id"],
            "q": q,
            "expect": it.get("expect") or it.get("expect_re"),
            "answer_preview": ans[:160],
            "stopped": r.stopped_reason,
            "ok": passed,
        })

    total = len(ITEMS)
    report = {
        "suite": "reasoning_tool_si_v1",
        "passed": ok_n,
        "total": total,
        "accuracy": round(ok_n / max(1, total), 4),
        "note": "Tool-augmented scores measure agent+calc reliability, not LM chain-of-thought.",
        "results": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nAccuracy: {report['accuracy']*100:.1f}% ({ok_n}/{total})")
    print(f"Saved {out}")
    return 0 if ok_n == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
