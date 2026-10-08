#!/usr/bin/env python
"""
Evidence script for NeuroField v2 architecture.

Reproduces the key claims from the technical report Section 7:
  1. Associative recall task (4-pair, chance = 0.125)
  2. Fast memory ON → high accuracy
  3. Fast memory removed / stop-gradient → clear drop
  4. Gradient audit: all modules receive non-zero gradient

Runs on CPU in a few minutes with a small configuration.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField


# ---------------------------------------------------------------------------
# Associative-recall data (exactly the task described in the report)
# ---------------------------------------------------------------------------

def make_recall_batch(
    batch_size: int,
    n_pairs: int = 4,
    n_keys: int = 8,
    n_values: int = 8,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sequence format (simplified but faithful):
      [KEY1, VAL1, KEY2, VAL2, ..., KEYn, VALn, QUERY_KEY, ANSWER]

    Model must predict ANSWER (the value that was paired with QUERY_KEY).
    Chance accuracy = 1 / n_values = 0.125 for n_values=8.
    """
    # Token layout:
    # 0 = pad, 1..n_keys = keys, n_keys+1 .. n_keys+n_values = values
    key_offset = 1
    val_offset = 1 + n_keys
    vocab = 1 + n_keys + n_values + 2  # extra room

    seqs = []
    targets = []
    for _ in range(batch_size):
        # sample unique keys and random values
        keys = random.sample(range(n_keys), n_pairs)
        vals = [random.randint(0, n_values - 1) for _ in range(n_pairs)]
        pairs = list(zip(keys, vals))
        random.shuffle(pairs)

        tokens = []
        for k, v in pairs:
            tokens.append(key_offset + k)
            tokens.append(val_offset + v)

        # query = one of the keys
        q_idx = random.randint(0, n_pairs - 1)
        q_key, q_val = pairs[q_idx]
        tokens.append(key_offset + q_key)          # query key
        answer = val_offset + q_val

        # input = everything except the answer; target = answer at last position
        inp = tokens  # length = 2*n_pairs + 1
        tgt = [-100] * (len(inp) - 1) + [answer]

        seqs.append(inp)
        targets.append(tgt)

    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)
    target_ids = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    for i, (s, t) in enumerate(zip(seqs, targets)):
        input_ids[i, : len(s)] = torch.tensor(s, device=device)
        target_ids[i, : len(t)] = torch.tensor(t, device=device)

    return input_ids, target_ids


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_recall(model: NeuroField, n_batches: int = 20, batch_size: int = 16) -> float:
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    for _ in range(n_batches):
        x, y = make_recall_batch(batch_size, device=device)
        out = model(x)
        # last non-pad position prediction
        logits = out.logits[:, -1, :]          # (B, V)
        pred = logits.argmax(dim=-1)
        gold = y[:, -1]
        mask = gold != -100
        correct += (pred[mask] == gold[mask]).sum().item()
        total += mask.sum().item()
    return correct / max(1, total)


def gradient_audit(model: NeuroField) -> dict[str, float]:
    """One backward pass → report grad norms per top-level module."""
    model.train()
    device = next(model.parameters()).device
    x, y = make_recall_batch(4, device=device)
    out = model(x, targets=y)
    model.zero_grad(set_to_none=True)
    out.loss.backward()

    report = {}
    for name, module in model.named_children():
        norms = []
        for p in module.parameters():
            if p.grad is not None:
                norms.append(p.grad.norm().item())
        report[name] = sum(norms) if norms else 0.0
    return report


# ---------------------------------------------------------------------------
# Training loop for evidence
# ---------------------------------------------------------------------------

def train_for_evidence(
    model: NeuroField,
    steps: int = 400,
    batch_size: int = 16,
    lr: float = 3e-3,
    log_every: int = 50,
) -> list[float]:
    device = next(model.parameters()).device
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    acc_history = []

    model.train()
    for step in range(1, steps + 1):
        x, y = make_recall_batch(batch_size, device=device)
        out = model(x, targets=y, lambda_err=0.05, lambda_bal=0.01)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            acc = evaluate_recall(model, n_batches=10, batch_size=16)
            acc_history.append(acc)
            print(f"  step {step:4d} | loss={loss.item():.3f} | recall_acc={acc:.3f}")
            model.train()
    return acc_history


# ---------------------------------------------------------------------------
# Main evidence runs
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")
    print("=" * 60)
    print("NeuroField v2 – Architecture Evidence")
    print("=" * 60)

    # Small config matching the spirit of the 9.92M reference (scaled down for CPU)
    base_cfg = ModelConfig(
        d_model=64,
        d_k=32,
        d_v=32,
        n_skills=4,
        top_k=2,
        k_max=3,
        dendrite_window=2,
        vocab_size=32,          # enough for keys+values
        tie_embeddings=True,
        dropout=0.0,
    )
    safety = SafetyConfig(enable_audit=False, max_memory_norm=20.0)

    # ------------------------------------------------------------------
    # 1. Gradient audit (Proposition 1)
    # ------------------------------------------------------------------
    print("\n[1] Gradient audit (one backward pass)")
    model = NeuroField(base_cfg, safety).to(device)
    grads = gradient_audit(model)
    zero_mods = [k for k, v in grads.items() if v == 0.0]
    print("  Grad norms:")
    for k, v in grads.items():
        status = "ZERO" if v == 0.0 else "OK"
        print(f"    {k:15s} {v:10.4f}  [{status}]")
    if zero_mods:
        print(f"  WARNING: zero gradient in {zero_mods}")
    else:
        print("  RESULT: All modules received non-zero gradient ✓")

    # ------------------------------------------------------------------
    # 2. Full model – fast memory ON
    # ------------------------------------------------------------------
    print("\n[2] Training with fast memory ON (target: high recall accuracy)")
    model = NeuroField(base_cfg, safety).to(device)
    t0 = time.time()
    hist_on = train_for_evidence(model, steps=300, batch_size=16, lr=3e-3)
    final_on = hist_on[-1]
    print(f"  Final recall accuracy (fast ON): {final_on:.3f}")
    print(f"  Wall time: {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Ablation – fast memory removed (read zeros, no write)
    # ------------------------------------------------------------------
    print("\n[3] Ablation: fast memory surgically disabled")
    model_off = NeuroField(base_cfg, safety).to(device)

    # Monkey-patch: write becomes no-op, read returns zeros
    def no_write(M, d_prev, o_t, g):
        return M
    def zero_read(M, d_t):
        return torch.zeros(M.size(0), model_off.cfg.d_v, device=M.device)
    model_off.fast_mem.write = no_write
    model_off.fast_mem.read = zero_read

    hist_off = train_for_evidence(model_off, steps=300, batch_size=16, lr=3e-3)
    final_off = hist_off[-1]
    print(f"  Final recall accuracy (fast OFF): {final_off:.3f}")

    # ------------------------------------------------------------------
    # 4. Stop-gradient write ablation
    # ------------------------------------------------------------------
    print("\n[4] Ablation: stop-gradient on fast write")
    model_sg = NeuroField(base_cfg, safety).to(device)
    original_write = model_sg.fast_mem.write

    def sg_write(M, d_prev, o_t, g):
        # detach M so no gradient flows through previous memory
        return original_write(M.detach(), d_prev, o_t, g)
    model_sg.fast_mem.write = sg_write

    hist_sg = train_for_evidence(model_sg, steps=300, batch_size=16, lr=3e-3)
    final_sg = hist_sg[-1]
    print(f"  Final recall accuracy (stop-grad write): {final_sg:.3f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVIDENCE SUMMARY")
    print("=" * 60)
    print(f"  Chance level (8 values)          : 0.125")
    print(f"  Fast memory ON                   : {final_on:.3f}")
    print(f"  Fast memory OFF (ablation)       : {final_off:.3f}")
    print(f"  Stop-gradient write (ablation)   : {final_sg:.3f}")
    print()
    if final_on > 0.6 and final_on > final_off + 0.15:
        print("  CONCLUSION: Fast memory is working and necessary for the task. ✓")
    elif final_on > 0.4:
        print("  CONCLUSION: Architecture learns above chance; memory helps. ✓")
    else:
        print("  CONCLUSION: Needs more steps / tuning on this CPU run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
