#!/usr/bin/env python3
"""Daily eval: append held-out metrics for a checkpoint to a JSONL log."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from neurofield.checkpoint.resolve import load_model_from_checkpoint
from neurofield.eval.suite import run_eval_suite, save_eval_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-data", required=True, help="held-out text file (never trained on)")
    ap.add_argument("--train-data", default=None)
    ap.add_argument("--identity-marker", default=None)
    ap.add_argument("--log-dir", default="docs/logs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, tok, _cfg, meta = load_model_from_checkpoint(args.checkpoint, None, device=args.device)
    val = Path(args.val_data).read_text(encoding="utf-8")
    train = Path(args.train_data).read_text(encoding="utf-8") if args.train_data else None
    marker = args.identity_marker or meta.get("model_id")

    report = run_eval_suite(
        model, tok.encode, tok.decode, text=train, val_text=val,
        device=args.device, identity_marker=marker,
    )
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    day = date.today().isoformat()
    report["date"] = day
    report["checkpoint"] = args.checkpoint
    out = log_dir / f"eval_{day}.json"
    save_eval_report(report, out)
    with (log_dir / "daily_eval.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": day, "val_nll": report["val_nll"], "identity_hit": report["identity_hit"]}) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:1200])
    print("Wrote", out)


if __name__ == "__main__":
    main()
