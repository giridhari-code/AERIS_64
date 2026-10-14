#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sentinel.agent.si_v1 import SI_v1Config, run_si_v1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    q = args.question or sys.stdin.read()
    r = run_si_v1(q, config=SI_v1Config(max_steps=args.max_steps))
    if args.json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(r.answer)
        print(f"[-- {r.stopped_reason} steps={len(r.steps)} ms={r.latency_ms:.0f}]", file=sys.stderr)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
