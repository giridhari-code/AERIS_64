"""0z code CLI — company-standard entrypoint.

Usage:
  python -m sentinel.zeroz.cli "What is the weather in Delhi?"
  python -m sentinel.zeroz.cli --config configs/zeroz.yaml -i
  python -m sentinel.zeroz.cli --policy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from sentinel.zeroz.config import load_zeroz_config
from sentinel.zeroz.policy import PRODUCT_NAME, PRODUCT_TAGLINE, SYSTEM_PROMPT
from sentinel.zeroz.runtime import build_generate_fn_from_checkpoint, run_zeroz
from sentinel.zeroz.session import ZeroZSession


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="0z",
        description=f"{PRODUCT_NAME} — {PRODUCT_TAGLINE}",
    )
    p.add_argument("-c", "--config", default=None, help="configs/zeroz.yaml")
    p.add_argument("--checkpoint", default=None, help="optional SENTINEL export dir")
    p.add_argument("--device", default=None)
    p.add_argument("--audit-dir", default=None)
    p.add_argument("--no-model", action="store_true", help="tools + templates only")
    p.add_argument("--metrics", action="store_true")
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--policy", action="store_true", help="print system policy and exit")
    p.add_argument("message", nargs="*", default=None, help="user message words")

    args = p.parse_args(argv)
    _setup_logging("DEBUG" if args.verbose else "WARNING")

    if args.policy:
        print(SYSTEM_PROMPT)
        raise SystemExit(0)

    cfg = load_zeroz_config(args.config)
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    if args.device:
        cfg.device = args.device
    if args.audit_dir:
        cfg.audit_dir = args.audit_dir
    if args.no_model:
        cfg.use_model_generate = False

    gen = None
    if cfg.use_model_generate and cfg.checkpoint:
        gen = build_generate_fn_from_checkpoint(cfg.checkpoint, device=cfg.device)

    session = ZeroZSession()
    text = " ".join(args.message).strip() if args.message else ""

    if not text and not args.interactive:
        p.print_help()
        raise SystemExit(2)

    def _one(msg: str) -> None:
        result = run_zeroz(msg, cfg=cfg, session=session, generate_fn=gen)
        print(result.answer)
        if args.metrics:
            print("---", file=sys.stderr)
            print(json.dumps(result.metrics(), indent=2), file=sys.stderr)

    if args.interactive:
        print(f"{PRODUCT_NAME} — {PRODUCT_TAGLINE}")
        print("Type quit/exit to leave. Coding requests are refused by policy.\n")
        if text:
            _one(text)
        while True:
            try:
                line = input("0z> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            _one(line)
        raise SystemExit(0)

    _one(text)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
