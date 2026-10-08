#!/usr/bin/env python3
"""
Production contribution ablation for NeuroField v2.

Uses the real neurofield.model.NeuroField (Appendix A), not a re-implementation.
Measures eval accuracy on 4-pair associative recall while switching modules via
AblationConfig (Section 7 protocol).

Presets:
  full              all modules on
  no_fast_mem       use_fast_memory=False
  no_slow_mem       use_slow_memory=False
  no_neuromod       use_neuromodulator=False (fixed_gate=0.5)
  no_metacog        use_metacognition=False (fixed_k=1)
  no_fast_no_slow   both memories off
  skills_only       fast+slow+neuromod+meta off

Example:
  PYTHONPATH=src python scripts/contribution_ablation.py \\
      --steps 800 --seeds 0,1,2 --device cpu --out runs/contrib.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig
from neurofield.model import NeuroField


PRESETS: dict[str, AblationConfig] = {
    "full": AblationConfig(),
    "no_fast_mem": AblationConfig(use_fast_memory=False),
    "no_slow_mem": AblationConfig(use_slow_memory=False),
    "no_neuromod": AblationConfig(use_neuromodulator=False, fixed_gate=0.5),
    "no_metacog": AblationConfig(use_metacognition=False, fixed_k=1),
    "no_fast_no_slow": AblationConfig(use_fast_memory=False, use_slow_memory=False),
    "skills_only": AblationConfig(
        use_fast_memory=False,
        use_slow_memory=False,
        use_neuromodulator=False,
        use_metacognition=False,
        fixed_gate=0.0,
        fixed_k=1,
    ),
}


def make_recall_batch(
    batch: int,
    n_pairs: int,
    n_keys: int,
    n_vals: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Sequence: k0 v0 k1 v1 ... k_{n-1} v_{n-1} QUERY_MARK key_q
    Target: only last position = value bound to key_q (ignore_index elsewhere).
    Token layout: 0 pad, 1 query mark, 2.. keys, then values.
    """
    key_offset = 2
    val_offset = key_offset + n_keys
    seqs: list[list[int]] = []
    tgts: list[list[int]] = []
    for _ in range(batch):
        keys = random.sample(range(n_keys), n_pairs)
        vals = [random.randrange(n_vals) for _ in range(n_pairs)]
        tokens: list[int] = []
        for k, v in zip(keys, vals):
            tokens.append(key_offset + k)
            tokens.append(val_offset + v)
        q = random.randrange(n_pairs)
        tokens.append(1)
        tokens.append(key_offset + keys[q])
        answer = val_offset + vals[q]
        tgt = [-100] * (len(tokens) - 1) + [answer]
        seqs.append(tokens)
        tgts.append(tgt)
    L = max(len(s) for s in seqs)
    x = torch.zeros(batch, L, dtype=torch.long, device=device)
    y = torch.full((batch, L), -100, dtype=torch.long, device=device)
    for i, (s, t) in enumerate(zip(seqs, tgts)):
        x[i, : len(s)] = torch.tensor(s, device=device)
        y[i, : len(t)] = torch.tensor(t, device=device)
    return x, y


def accuracy_supervised(logits: Tensor, targets: Tensor) -> float:
    pred = logits.argmax(dim=-1)
    mask = targets != -100
    if int(mask.sum()) == 0:
        return 0.0
    return float((pred[mask] == targets[mask]).float().mean().item())


def run_one(
    name: str,
    ablation: AblationConfig,
    steps: int,
    seed: int,
    d_model: int,
    batch: int,
    n_pairs: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)

    n_keys, n_vals = 8, 8
    vocab = 2 + n_keys + n_vals
    cfg = ModelConfig(
        d_model=d_model,
        d_k=max(16, d_model // 2),
        d_v=max(16, d_model // 2),
        n_skills=4,
        top_k=2,
        k_max=3,
        dendrite_window=3,
        vocab_size=vocab,
        max_seq_len=64,
        tie_embeddings=True,
        dropout=0.0,
    )
    safety = SafetyConfig(
        enable_audit=False,
        max_write_norm=10.0,
        max_memory_norm=50.0,
        gate_cap=1.0,
        truncate_write_window=32,
    )
    model = NeuroField(cfg, safety=safety, ablation=ablation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    best_acc = 0.0
    last_acc = 0.0
    last_ce = float("nan")

    model.train()
    for step in range(1, steps + 1):
        x, y = make_recall_batch(batch, n_pairs, n_keys, n_vals, device)
        out = model(x, targets=y, lambda_err=0.1, lambda_bal=0.01)
        assert out.loss is not None
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Paper: surprise-weighted replay into slow memory every 50–100 steps
        if (
            ablation.use_slow_memory
            and out.state is not None
            and step % 50 == 0
            and "_seq_surprise" in out.metrics
        ):
            model.replay_slow(out.state["M"], out.metrics["_seq_surprise"], eta=0.02)
        last_acc = accuracy_supervised(out.logits.detach(), y)
        last_ce = float(out.metrics.get("ce", out.loss.item()))
        best_acc = max(best_acc, last_acc)

    model.eval()
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(32):
            x, y = make_recall_batch(batch, n_pairs, n_keys, n_vals, device)
            out = model(x, targets=y, lambda_err=0.0, lambda_bal=0.0)
            accs.append(accuracy_supervised(out.logits, y))
    eval_acc = sum(accs) / len(accs)

    return {
        "ablation": name,
        "seed": seed,
        "steps": steps,
        "eval_acc": eval_acc,
        "train_best_acc": best_acc,
        "train_last_acc": last_acc,
        "train_last_ce": last_ce,
        "params": model.param_report()["total"],
        "wall_sec": round(time.time() - t0, 2),
        "flags": {
            "use_fast_memory": ablation.use_fast_memory,
            "use_slow_memory": ablation.use_slow_memory,
            "use_neuromodulator": ablation.use_neuromodulator,
            "use_metacognition": ablation.use_metacognition,
            "fixed_gate": ablation.fixed_gate,
            "fixed_k": ablation.fixed_k,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="NeuroField production contribution ablation")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--ablations",
        type=str,
        default="full,no_fast_mem,no_slow_mem,no_neuromod,no_metacog,no_fast_no_slow,skills_only",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs/contribution_ablation.json")
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    names = [n.strip() for n in args.ablations.split(",") if n.strip()]

    results: list[dict[str, Any]] = []
    print(f"device={device} steps={args.steps} seeds={seeds} d_model={args.d_model}")
    print(f"chance accuracy ({args.pairs}-pair / 8 vals) = {1 / 8:.3f}")
    print("-" * 72)

    for name in names:
        if name not in PRESETS:
            print(f"unknown ablation {name!r}, skip")
            continue
        rows = []
        for seed in seeds:
            row = run_one(
                name,
                PRESETS[name],
                args.steps,
                seed,
                args.d_model,
                args.batch,
                args.pairs,
                args.lr,
                device,
            )
            rows.append(row)
            results.append(row)
            print(
                f"{name:16s} seed={seed}  eval_acc={row['eval_acc']:.3f}  "
                f"best_train={row['train_best_acc']:.3f}  ce={row['train_last_ce']:.3f}  "
                f"params={row['params']}  wall={row['wall_sec']}s"
            )
        if len(rows) > 1:
            accs = [r["eval_acc"] for r in rows]
            print(
                f"{'':16s} mean={sum(accs) / len(accs):.3f}  "
                f"range=[{min(accs):.3f}, {max(accs):.3f}]"
            )
        print("-" * 72)

    print("\n=== Contribution summary (eval accuracy) ===")
    by: dict[str, list[float]] = {}
    for r in results:
        by.setdefault(r["ablation"], []).append(r["eval_acc"])
    full_mean = sum(by.get("full", [0.0])) / max(1, len(by.get("full", [1])))
    for name, accs in by.items():
        m = sum(accs) / len(accs)
        print(f"  {name:16s}  mean={m:.3f}  Δ vs full={m - full_mean:+.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out.resolve()}")


if __name__ == "__main__":
    main()
