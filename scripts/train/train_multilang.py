#!/usr/bin/env python3
"""Train NeuroField on India multilang + Hindlish + spelling-robust data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from neurofield.checkpoint import save_checkpoint
from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/india_multilang.txt")
    ap.add_argument("--extra", default="data/skills_real.txt", help="optional extra corpus")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="docs/neurofield_india")
    ap.add_argument("--d-model", type=int, default=64)
    args = ap.parse_args()

    parts = []
    for path in [args.data, args.extra]:
        p = Path(path)
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
            print(f"loaded {p} ({p.stat().st_size} bytes)")
    text = "\n".join(parts)
    # drop comment-only lines starting with # for cleaner char distribution optional
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    text = "\n".join(lines) + "\n"

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"chars={len(data)} vocab={len(chars)} (includes Devanagari if present)")

    cfg = ModelConfig(
        d_model=args.d_model,
        d_k=args.d_model // 2,
        d_v=args.d_model // 2,
        n_skills=4,
        top_k=2,
        k_max=2,
        dendrite_window=3,
        vocab_size=len(chars),
        tie_embeddings=True,
        dropout=0.05,
    )
    model = NeuroField(cfg, SafetyConfig(enable_audit=False, max_memory_norm=40.0))
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def batch(bs=16, seq=48):
        ix = torch.randint(0, max(1, len(data) - seq - 1), (bs,))
        x = torch.stack([data[i : i + seq] for i in ix])
        y = torch.stack([data[i + 1 : i + seq + 1] for i in ix])
        return x, y

    print(f"Training {args.steps} steps...")
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch()
        loss = model(x, targets=y, lambda_err=0.03, lambda_bal=0.005).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d} | loss={loss.item():.3f}")
    print(f"Done in {time.time() - t0:.1f}s")

    out = save_checkpoint(
        args.out,
        model,
        config={
            "d_model": cfg.d_model,
            "d_k": cfg.d_k,
            "d_v": cfg.d_v,
            "n_skills": cfg.n_skills,
            "top_k": cfg.top_k,
            "k_max": cfg.k_max,
            "dendrite_window": cfg.dendrite_window,
            "vocab_size": cfg.vocab_size,
            "tie_embeddings": True,
        },
        stoi=stoi,
        itos=itos,
        meta={
            "source": [args.data, args.extra],
            "steps": args.steps,
            "languages": ["en", "hindlish", "hi", "bn", "ta", "te", "mr", "gu"],
        },
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
