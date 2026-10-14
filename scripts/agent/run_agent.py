#!/usr/bin/env python3
"""CLI: reliable agent without or with a checkpoint generate_fn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neurofield.agent import AgentConfig, run_agent


def main() -> int:
    ap = argparse.ArgumentParser(description="AERIS reliable agent")
    ap.add_argument("question", nargs="?", default="")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    q = args.question or sys.stdin.read()
    cfg = AgentConfig(max_steps=args.max_steps)
    result = run_agent(q, generate_fn=None, config=cfg)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.answer)
        print(f"\n[-- stopped={result.stopped_reason} steps={result.steps} ms={result.latency_ms:.0f}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
