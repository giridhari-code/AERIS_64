#!/usr/bin/env python3
"""Evaluate a NeuroField checkpoint on HELD-OUT text.

--val-data must be text the model never trained on. Without it, validation
metrics are reported as null (instead of the old train-set split that made
perplexity look great).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurofield.checkpoint.resolve import load_model_from_checkpoint
from neurofield.eval.suite import run_eval_suite, save_eval_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="docs/neurofield_company")
    ap.add_argument("--config", default=None, help="optional YAML (non-shape knobs only)")
    ap.add_argument("--train-data", default=None, help="text used for training (train_nll only)")
    ap.add_argument("--val-data", default=None, help="held-out text (REQUIRED for val metrics)")
    ap.add_argument("--identity-marker", default=None, help="string expected in the 'who am I' reply")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok, _cfg, _meta = load_model_from_checkpoint(args.checkpoint, args.config, device=args.device)

    def _read(p):
        return Path(p).read_text(encoding="utf-8") if p and Path(p).is_file() else None

    report = run_eval_suite(
        model, tok.encode, tok.decode,
        text=_read(args.train_data), val_text=_read(args.val_data),
        device=args.device, identity_marker=args.identity_marker,
    )
    out = args.out or str(Path(args.checkpoint) / "eval_report.json")
    save_eval_report(report, out)
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}, indent=2))
    print("generations:")
    for k, v in report.get("generations", {}).items():
        print(f"  [{k}] => {v[:120]!r}")


if __name__ == "__main__":
    main()
