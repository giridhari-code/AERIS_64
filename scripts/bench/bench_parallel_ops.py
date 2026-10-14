"""Benchmark the token-parallel forward path against the per-token path (fwd+bwd, CPU or CUDA).

    python scripts/bench/bench_parallel_ops.py --d-model 128 --vocab 8000 --seq 128 --batch 8
"""
from __future__ import annotations

import argparse
import time

import torch

from sentinel.config import ModelConfig, SafetyConfig
from sentinel.model import SENTINEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skills", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cfg = ModelConfig(d_model=a.d_model, d_k=32, d_v=32, n_skills=a.skills, top_k=2, k_max=2,
                      vocab_size=a.vocab, dendrite_window=3, max_seq_len=a.seq, dropout=0.0)
    torch.manual_seed(0)
    m = SENTINEL(cfg, SafetyConfig(enable_audit=False)).to(dev).train()
    x = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
    y = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
    res = {}
    for name, flag in (("per-token (legacy)", False), ("token-parallel", True)):
        m.parallel_ops = flag
        for _ in range(2):  # warmup
            m(x, targets=y).loss.backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            m.zero_grad(set_to_none=True)
            m(x, targets=y).loss.backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        res[name] = (time.perf_counter() - t0) / a.iters
        print(f"{name:20s} {res[name]*1000:9.1f} ms / step  ({a.batch*a.seq/res[name]:,.0f} tok/s)")
    print(f"speed-up: {res['per-token (legacy)'] / res['token-parallel']:.2f}x")


if __name__ == "__main__":
    main()
