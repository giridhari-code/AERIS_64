#!/usr/bin/env python3
"""Micro-benchmark: NeuroField vs minimal Transformer block (same d_model, seq).

Measures wall time + rough param counts — for cost intuition, not paper claims.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurofield.config import ModelConfig, SafetyConfig, model_preset
from neurofield.model import NeuroField


class MiniTransformerLM(nn.Module):
    """Minimal GPT-style block stack for fair-ish timing baseline."""

    def __init__(self, vocab: int, d_model: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, n_heads),
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        # causal mask
        t = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def bench(fn, warmup=3, runs=10) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["tiny", "small", "medium"])
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    cfg = model_preset(args.preset)
    cfg.vocab_size = args.vocab
    cfg.d_k = max(8, cfg.d_model // 2)
    cfg.d_v = max(8, cfg.d_model // 2)

    nf = NeuroField(cfg, SafetyConfig(enable_audit=False)).to(device)
    # match param ballpark roughly: 2 layers
    tf = MiniTransformerLM(args.vocab, cfg.d_model, n_layers=2, n_heads=max(1, cfg.d_model // 16)).to(device)

    x = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
    y = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)

    def nf_fwd():
        with torch.no_grad():
            nf(x)

    def tf_fwd():
        with torch.no_grad():
            tf(x)

    def nf_train_step():
        nf.train()
        out = nf(x, targets=y)
        loss = out.loss if out.loss is not None else F.cross_entropy(
            out.logits.reshape(-1, out.logits.size(-1)), y.reshape(-1)
        )
        loss.backward()
        nf.zero_grad(set_to_none=True)

    def tf_train_step():
        tf.train()
        logits = tf(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        tf.zero_grad(set_to_none=True)

    print("=== Cost micro-benchmark (wall time) ===")
    print(f"preset={args.preset} d_model={cfg.d_model} seq={args.seq} batch={args.batch} device={device}")
    print(f"NeuroField params:     {count_params(nf):,}")
    print(f"MiniTransformer params:{count_params(tf):,}")
    print(f"NF  forward ms: {bench(nf_fwd)*1000:.2f}")
    print(f"TF  forward ms: {bench(tf_fwd)*1000:.2f}")
    print(f"NF  train  ms: {bench(nf_train_step)*1000:.2f}")
    print(f"TF  train  ms: {bench(tf_train_step)*1000:.2f}")
    print("Note: MiniTransformer is a simple baseline, not GPT-4. Use for local cost intuition only.")


if __name__ == "__main__":
    main()
