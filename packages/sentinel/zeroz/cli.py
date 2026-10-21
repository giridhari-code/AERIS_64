"""0z code CLI — Claude Code–similar interactive agent (no coding).

Usage:
  python -m sentinel.zeroz.cli -i
  python -m sentinel.zeroz.cli --no-model "Weather in Delhi?"
  python -m sentinel.zeroz.cli --policy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from sentinel.zeroz.config import load_zeroz_config
from sentinel.zeroz.context import load_project_context
from sentinel.zeroz.policy import PRODUCT_NAME, PRODUCT_TAGLINE, SYSTEM_PROMPT
from sentinel.zeroz.runtime import build_generate_fn_from_checkpoint, run_zeroz
from sentinel.zeroz.session import ZeroZSession
from sentinel.zeroz.todos import TodoList
from sentinel.zeroz.toolbus import ToolBus


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _print_help_interactive() -> None:
    print(
        """
Slash commands (Claude Code–similar):
  /help       this help
  /tools      list allow-listed tools
  /policy     product policy
  /context    project root + key files
  /todos      show session todos
  /todo add <text>
  /todo done <id>
  /plan <goal>   show tool plan only (no execute)
  /clear      clear session transcript
  /quit       exit
""".strip()
    )


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="0z",
        description=f"{PRODUCT_NAME} — {PRODUCT_TAGLINE} (Claude Code–similar, zero coding)",
    )
    p.add_argument("-c", "--config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--audit-dir", default=None)
    p.add_argument("--no-model", action="store_true")
    p.add_argument("--metrics", action="store_true")
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--policy", action="store_true")
    p.add_argument("--context", action="store_true", help="print project context and exit")
    p.add_argument("message", nargs="*", default=None)

    args = p.parse_args(argv)
    _setup_logging("DEBUG" if args.verbose else "WARNING")

    if args.policy:
        print(SYSTEM_PROMPT)
        raise SystemExit(0)
    if args.context:
        print(load_project_context())
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
    todos = TodoList()
    text = " ".join(args.message).strip() if args.message else ""

    def _one(msg: str) -> None:
        result = run_zeroz(msg, cfg=cfg, session=session, generate_fn=gen)
        print(result.answer)
        if args.metrics:
            print("---", file=sys.stderr)
            print(json.dumps(result.metrics(), indent=2), file=sys.stderr)

    def _handle_slash(line: str) -> bool:
        """Return True if handled as slash command."""
        if not line.startswith("/"):
            return False
        parts = line.split(maxsplit=2)
        cmd = parts[0].lower()
        if cmd in ("/quit", "/exit", "/q"):
            raise SystemExit(0)
        if cmd in ("/help", "/?"):
            _print_help_interactive()
            return True
        if cmd == "/tools":
            bus = ToolBus(allowed=set(cfg.allowed_tools))
            for s in bus.schemas():
                print(f"  - {s['name']}: {s['description']}")
            return True
        if cmd == "/policy":
            print(SYSTEM_PROMPT)
            return True
        if cmd == "/context":
            print(load_project_context())
            return True
        if cmd == "/todos":
            print(todos.render())
            return True
        if cmd == "/todo":
            if len(parts) < 2:
                print("usage: /todo add <text> | /todo done <id>")
                return True
            sub = parts[1].lower()
            rest = parts[2] if len(parts) > 2 else ""
            if sub == "add" and rest:
                it = todos.add(rest)
                print(f"added #{it.id}")
            elif sub == "done" and rest.isdigit():
                ok = todos.set_status(int(rest), "done")
                print("ok" if ok else "not found")
            else:
                print("usage: /todo add <text> | /todo done <id>")
            return True
        if cmd == "/plan":
            from sentinel.zeroz.planner import build_plan

            goal = line[len("/plan") :].strip() or text
            plan = build_plan(goal)
            print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
            return True
        if cmd == "/clear":
            session.turns.clear()
            print("session cleared")
            return True
        print(f"unknown command {cmd}. /help")
        return True

    if not text and not args.interactive:
        p.print_help()
        raise SystemExit(2)

    if args.interactive:
        print(f"{PRODUCT_NAME} — {PRODUCT_TAGLINE}")
        print("Claude Code–similar agent · coding disabled by policy")
        print("Type /help for slash commands. quit/exit to leave.\n")
        if text:
            if not _handle_slash(text):
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
            if _handle_slash(line):
                continue
            _one(line)
        raise SystemExit(0)

    if text.startswith("/"):
        _handle_slash(text)
    else:
        _one(text)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
