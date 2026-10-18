#!/usr/bin/env python3
"""SI_v0 CLI: reliable agent (± checkpoint generate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from sentinel.agent import AgentConfig, run_agent
from sentinel.agent.generate import make_generate_fn


def main() -> int:
    ap = argparse.ArgumentParser(description="AERIS SI_v0 reliable agent")
    ap.add_argument("question", nargs="?", default="")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--checkpoint", default="", help="Optional SENTINEL ckpt for generate_fn")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    q = args.question or sys.stdin.read()
    gen = None
    if args.checkpoint:
        gen = make_generate_fn(
            args.checkpoint,
            device=args.device,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
        )
    cfg = AgentConfig(max_steps=args.max_steps)
    result = run_agent(q, generate_fn=gen, config=cfg)
    if gen is not None and getattr(gen, "load_error", None):
        print(f"[-- generate load_error: {gen.load_error}]", file=sys.stderr)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.answer)
        print(
            f"\n[-- stopped={result.stopped_reason} steps={result.steps} ms={result.latency_ms:.0f}]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
