#!/usr/bin/env python3
"""
Ablations for NeuroField memory fixes:
  - target.detach() on prediction error (default on)
  - multi-timescale decay init (linspace log_lambda)

Runs a short synthetic next-token step to verify forward/backward still work
and prints decay timescales + a one-step write-norm shape check.
"""

from __future__ import annotations

import argparse

import torch

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField
from neurofield.modules.memory import FastMemory
from neurofield.modules.predictor import Predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--d-k", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--no-detach-target", action="store_true")
    args = ap.parse_args()

    cfg = ModelConfig(
        d_model=args.d_model,
        d_k=args.d_k,
        d_v=args.d_k,
        n_skills=4,
        top_k=2,
        k_max=2,
        dendrite_window=3,
        vocab_size=64,
        max_seq_len=64,
        tie_embeddings=True,
    )
    model = NeuroField(cfg, SafetyConfig(enable_audit=True, max_write_norm=5.0))
    model.train()

    # Decay timescales
    lam = model.fast_mem.lambda_diag.detach()
    print("lambda_diag (decay per step):", [round(float(x), 4) for x in lam.tolist()])
    print("  half-life-ish tokens (~ln0.5/lnλ) for slowest/fastest:")
    for name, v in [("min", lam.min()), ("max", lam.max())]:
        import math
        lv = float(v)
        hl = math.log(0.5) / math.log(lv) if 0 < lv < 1 else float("inf")
        print(f"  {name} λ={lv:.4f} → ~{hl:.1f} steps")

    B, T, V = args.batch, args.seq, cfg.vocab_size
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    out = model(x, targets=y, lambda_err=0.05, lambda_bal=0.0)
    assert out.loss is not None
    out.loss.backward()
    print("forward+backward OK | loss=", float(out.loss))

    # Predictor detach check
    pred = torch.randn(B, args.d_model, requires_grad=True)
    target = torch.randn(B, args.d_model, requires_grad=True)
    err, S = Predictor.compute_error_and_surprise(pred, target, detach_target=not args.no_detach_target)
    err.pow(2).mean().backward()
    print("target.grad is None (want True when detach):", target.grad is None)
    print("pred.grad is not None (want True):", pred.grad is not None)

    # Per-seq write norm shape
    fm = FastMemory(args.d_model, args.d_k, args.d_k)
    M = torch.zeros(B, args.d_k, args.d_k)
    d = torch.randn(B, args.d_model)
    o = torch.randn(B, args.d_model)
    g = torch.ones(B, 1)
    M2 = fm.write(M, d, o, g)
    wn = (M2 - M).norm(dim=(-2, -1))
    print("write norm shape (want [B]):", tuple(wn.shape))
    print("ablation script done")


if __name__ == "__main__":
    main()
