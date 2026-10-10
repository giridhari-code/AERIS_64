#!/usr/bin/env python3
"""
Fine-tuning entry (SFT-style): continue train from a checkpoint on domain / chat data.

Wraps the company trainer flags for a clear "finetune" workflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune AERIS from checkpoint")
    ap.add_argument("--resume", required=True, help="base checkpoint folder")
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--cosine", action="store_true", default=True)
    args, extra = ap.parse_known_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train" / "train_company.py"),
        "--resume",
        args.resume,
        "--data",
        *args.data,
        "--out",
        args.out,
        "--steps",
        str(args.steps),
        "--preset",
        args.preset,
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--grad-accum",
        str(args.grad_accum),
        "--warmup",
        str(args.warmup),
    ]
    if args.cosine:
        cmd.append("--cosine")
    cmd.extend(extra)
    print("fine-tune →", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
