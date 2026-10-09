#!/usr/bin/env python3
"""Build a train-ready text corpus for AERIS / NeuroField.

Default (local / CPU-friendly):
  - data/samples/*.txt
  - data/align/*.txt (if present)
  - synthetic chat (~100k chars by default)

Optional:
  --extra path1 path2   more .txt files
  --synthetic-chars N   0 to skip synthetic
  --bpe                 also run BPE packing via prepare_data.py (needs tokenizers)

Outputs:
  data/ready/train.txt
  data/ready/manifest.json

Usage:
  PYTHONPATH=src python scripts/data/build_dataset.py
  PYTHONPATH=src python scripts/data/build_dataset.py --synthetic-chars 200000
  PYTHONPATH=src python scripts/data/build_dataset.py --bpe --vocab-size 4000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AERIS train corpus")
    ap.add_argument("--out-dir", default="data/ready")
    ap.add_argument("--synthetic-chars", type=int, default=100_000,
                    help="synthetic chat budget (0 = skip)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra .txt paths")
    ap.add_argument("--bpe", action="store_true", help="also tokenize with prepare_data.py")
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"

    sources: list[tuple[str, int]] = []
    parts: list[str] = []

    # 1) samples
    samples = sorted((ROOT / "data" / "samples").glob("*.txt"))
    for p in samples:
        t = read_text(p)
        if not t:
            continue
        parts.append(t)
        sources.append((str(p.relative_to(ROOT)), len(t)))

    # 2) align
    align_dir = ROOT / "data" / "align"
    if align_dir.is_dir():
        for p in sorted(align_dir.glob("*.txt")):
            t = read_text(p)
            if not t:
                continue
            parts.append(t)
            sources.append((str(p.relative_to(ROOT)), len(t)))

    # 3) synthetic
    if args.synthetic_chars > 0:
        syn_script = ROOT / "scripts" / "data" / "make_synthetic_100k.py"
        syn_out = out_dir / "synthetic_chat.txt"
        cmd = [
            sys.executable,
            str(syn_script),
            "--out", str(syn_out),
            "--target-chars", str(args.synthetic_chars),
            "--seed", str(args.seed),
        ]
        subprocess.check_call(cmd)
        t = read_text(syn_out)
        parts.append(t)
        sources.append((str(syn_out.relative_to(ROOT)), len(t)))

    # 4) extra
    for e in args.extra:
        p = Path(e)
        if not p.is_file():
            print(f"skip missing extra: {p}", file=sys.stderr)
            continue
        t = read_text(p)
        parts.append(t)
        sources.append((str(p), len(t)))

    if not parts:
        print("ERROR: no text sources found", file=sys.stderr)
        sys.exit(1)

    # Separate with blank lines between files
    body = "\n\n".join(parts) + "\n"
    train_path.write_text(body, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_txt": str(train_path.relative_to(ROOT)),
        "total_chars": len(body),
        "approx_words": len(body.split()),
        "sources": [{"path": s, "chars": c} for s, c in sources],
        "note": (
            "This is a small local corpus. For 1B pretrain you need multi-GB real text "
            "(billions of tokens), not this file alone."
        ),
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== dataset ready ===")
    print(f"train:     {train_path}")
    print(f"chars:     {len(body):,}")
    print(f"words≈     {manifest['approx_words']:,}")
    print(f"manifest:  {man_path}")
    for s, c in sources:
        print(f"  + {c:8,}  {s}")

    if args.bpe:
        prep = ROOT / "scripts" / "data" / "prepare_data.py"
        tok_dir = out_dir / "tokens"
        cmd = [
            sys.executable,
            str(prep),
            "--input", str(train_path),
            "--out-dir", str(tok_dir),
            "--vocab-size", str(args.vocab_size),
        ]
        print("running BPE prepare_data:", " ".join(cmd))
        subprocess.check_call(cmd)
        print(f"tokens dir: {tok_dir}")


if __name__ == "__main__":
    main()
