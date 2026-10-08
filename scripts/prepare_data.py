#!/usr/bin/env python3
"""Tokenize raw text into train.bin / val.bin (uint16) for 1B-scale training.

    pip install tokenizers numpy
    python scripts/prepare_data.py --input "data/raw/*.txt" --out-dir data/tokens \
        --vocab-size 32000 --val-frac 0.005

What it guarantees (the old pipeline did none of this):
  * exact-duplicate chunks are dropped BEFORE splitting (no train/val duplicates)
  * the validation split is deterministic (hash of the chunk), written to its own file
  * the tokenizer is trained on the TRAIN side only (no vocab leakage)
  * every id is range-checked against the output dtype

Limitations: this does exact dedup only (no near-dup / MinHash), no language
or quality filtering, and no licence checks. Those matter more than model size;
see docs/TRAIN_1B.md.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

EOS = "<|endoftext|>"


def iter_chunks(files: list[str], chunk_chars: int):
    """Yield (file, chunk) of ~chunk_chars, cut on paragraph boundaries."""
    for f in files:
        buf, size = [], 0
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for para in fh.read().split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                buf.append(para)
                size += len(para)
                if size >= chunk_chars:
                    yield f, "\n\n".join(buf)
                    buf, size = [], 0
        if buf:
            yield f, "\n\n".join(buf)


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="files or globs of .txt")
    ap.add_argument("--out-dir", default="data/tokens")
    ap.add_argument("--tokenizer", default=None, help="existing HF tokenizer.json (skip training)")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--split-by", choices=["chunk", "file"], default="chunk",
                    help="'file' keeps whole files on one side (safer when files are documents)")
    ap.add_argument("--chunk-chars", type=int, default=20000)
    ap.add_argument("--dtype", choices=["uint16", "int32"], default="uint16")
    args = ap.parse_args()

    files = sorted({f for pat in args.input for f in glob.glob(pat)})
    if not files:
        sys.exit("no input files matched")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_mille = max(1, int(args.val_frac * 1000))

    # pass 1: dedupe + split into two temp text files
    seen: set[int] = set()
    dup = n_train = n_val = 0
    with tempfile.TemporaryDirectory() as td:
        tr_path, va_path = Path(td) / "train.txt", Path(td) / "val.txt"
        with open(tr_path, "w", encoding="utf-8") as ftr, open(va_path, "w", encoding="utf-8") as fva:
            for f, chunk in iter_chunks(files, args.chunk_chars):
                hh = h64(chunk)
                if hh in seen:
                    dup += 1
                    continue
                seen.add(hh)
                key = h64(f) if args.split_by == "file" else hh
                if key % 1000 < per_mille:
                    fva.write(chunk + "\n\n<EOD>\n\n")
                    n_val += 1
                else:
                    ftr.write(chunk + "\n\n<EOD>\n\n")
                    n_train += 1
        print(f"chunks: train={n_train} val={n_val} duplicates_dropped={dup}")
        if n_val == 0:
            sys.exit("validation split is empty; raise --val-frac or add data")

        # tokenizer (train side only)
        tok_path = out / "tokenizer.json"
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError:
            sys.exit("pip install tokenizers")
        if args.tokenizer:
            tok = Tokenizer.from_file(args.tokenizer)
        else:
            tok = Tokenizer(models.BPE())
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=args.vocab_size,
                special_tokens=[EOS],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )
            tok.train([str(tr_path)], trainer)
        tok.save(str(tok_path))
        vocab = tok.get_vocab_size()
        limit = 65536 if args.dtype == "uint16" else 2**31
        if vocab > limit:
            sys.exit(f"vocab {vocab} does not fit {args.dtype}")
        eos_id = tok.token_to_id(EOS)
        print(f"tokenizer vocab={vocab} eos_id={eos_id} -> {tok_path}")

        # pass 2: encode
        np_dtype = np.uint16 if args.dtype == "uint16" else np.int32
        for name, src in (("train", tr_path), ("val", va_path)):
            n_tok = 0
            with open(out / f"{name}.bin", "wb") as fb:
                for doc in src.read_text(encoding="utf-8").split("\n\n<EOD>\n\n"):
                    if not doc.strip():
                        continue
                    ids = tok.encode(doc).ids + ([eos_id] if eos_id is not None else [])
                    arr = np.asarray(ids, dtype=np.int64)
                    assert arr.max() < limit
                    arr.astype(np_dtype).tofile(fb)
                    n_tok += len(ids)
            print(f"{name}: {n_tok:,} tokens -> {out / (name + '.bin')}")
    print("done. Use data.train_path/val_path in configs/aeris_1b.yaml and pass --tokenizer-json to neurofield.cli")


if __name__ == "__main__":
    main()
