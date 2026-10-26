#!/usr/bin/env python3
"""Colab / local end-to-end test: ~1M words → tokenize → short AERIS train.

Usage:
  cd AERIS_64
  pip install -e ".[dev]" tokenizers
  PYTHONPATH=packages python scripts/train/colab_1m_words_test.py

  # faster / CPU smoke:
  PYTHONPATH=packages python scripts/train/colab_1m_words_test.py --words 200000 --max-steps 50 --cpu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def make_corpus(path: Path, target_words: int) -> int:
    """Write unique-ish paragraphs so prepare_data dedup keeps enough chunks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    topics = [
        "two-speed memory", "slow ring slots", "neural field loops", "metacognition control",
        "hierarchical predictors", "prediction error", "skill routing", "j-space workspace",
        "bounded KV alternative", "surprise gated writes", "dendrite windows", "replay consolidation",
    ]
    words = 0
    n = 0
    with path.open("w", encoding="utf-8") as f:
        while words < target_words:
            topic = topics[n % len(topics)]
            # Unique paragraph per n — survives exact-hash dedup in prepare_data
            line = (
                f"AERIS document {n} about {topic}. "
                f"Token series unique id={n} hex={n:08x}. "
                f"The model learns that paragraph {n} mentions {topic} with detail index {n * 17}. "
                f"Secondary fact: value {n % 97} links to concept {(n * 13) % 101}. "
                f"Closing sentence for chunk {n} keeps training data diverse.\n\n"
            )
            f.write(line)
            words += len(line.split())
            n += 1
    return words


def run(cmd: list[str], env: dict) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="1M-words Colab pipeline test for AERIS_64")
    ap.add_argument("--words", type=int, default=1_000_000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=8000)
    args = ap.parse_args()

    raw = ROOT / "data" / "raw" / "colab_1m.txt"
    tok_dir = ROOT / "data" / "tokens_1m"
    cfg = ROOT / "configs" / "colab_1m_test.yaml"

    print(f"== corpus target ~{args.words:,} words → {raw}")
    t0 = time.time()
    nwords = make_corpus(raw, args.words)
    print(f"wrote {nwords:,} words in {time.time() - t0:.1f}s")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "packages") + os.pathsep + env.get("PYTHONPATH", "")

    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "data" / "prepare_data.py"),
            "--input",
            str(raw),
            "--out-dir",
            str(tok_dir),
            "--vocab-size",
            str(args.vocab_size),
            "--val-frac",
            "0.01",
            "--chunk-chars",
            "2000",
        ],
        env=env,
    )

    if args.skip_train:
        print("data ready:", tok_dir)
        return

    device = "cpu" if args.cpu else "cuda"
    precision = "fp32" if args.cpu else "bf16"
    train_cmd = [
        sys.executable,
        "-m",
        "sentinel.cli",
        "train",
        "--config",
        str(cfg),
        "--data-dir",
        str(tok_dir),
        "--max-steps",
        str(args.max_steps),
        "--device",
        device,
        "--precision",
        precision,
        "--tokenizer-json",
        str(tok_dir / "tokenizer.json"),
    ]
    if args.cpu:
        train_cmd.extend(["--batch-size", "2"])

    print("== train")
    run(train_cmd, env=env)
    print("== done → runs/colab_1m_test")


if __name__ == "__main__":
    main()
