#!/usr/bin/env python3
"""
Minimal experiment harness for AESC paper tables (honest small-scale).

Runs:
  1) Associative-recall smoke (train few steps, report accuracy)
  2) Optional next-token CE on a text file
  3) Optional TinyTransformer baseline CE on same file

Does NOT invent large-scale results.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from sentinel.config import ModelConfig, SafetyConfig
from sentinel.model import SENTINEL
from sentinel.baselines.transformer_lm import TinyTransformerLM


def associative_recall(steps: int = 200, device: str = "cpu") -> dict:
    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=64, d_k=32, d_v=32, n_skills=4, top_k=2, k_max=2,
        vocab_size=32, dendrite_window=3, max_seq_len=64, dropout=0.0,
    )
    model = SENTINEL(cfg, SafetyConfig(enable_audit=False)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def make_batch(B=8, pairs=2):
        # vocab: 1..8 keys, 9..16 values
        keys = torch.randint(1, 9, (B, pairs), device=device)
        vals = torch.randint(9, 17, (B, pairs), device=device)
        # sequence: k v k v ... probe_k answer
        T = pairs * 2 + 2
        x = torch.zeros(B, T, dtype=torch.long, device=device)
        y = torch.full((B, T), -100, dtype=torch.long, device=device)
        for b in range(B):
            pos = 0
            for i in range(pairs):
                x[b, pos] = keys[b, i]
                y[b, pos] = keys[b, i]
                pos += 1
                x[b, pos] = vals[b, i]
                y[b, pos] = vals[b, i]
                pos += 1
            # probe first key, target first val
            x[b, pos] = keys[b, 0]
            y[b, pos] = keys[b, 0]
            pos += 1
            x[b, pos] = 0  # placeholder
            y[b, pos] = vals[b, 0]
        return x, y

    model.train()
    last_loss = None
    for step in range(1, steps + 1):
        x, y = make_batch()
        opt.zero_grad(set_to_none=True)
        out = model(x, targets=y, lambda_err=0.01, lambda_bal=0.001)
        out.loss.backward()
        opt.step()
        last_loss = float(out.loss.detach())

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(20):
            x, y = make_batch(B=16)
            logits = model(x).logits
            pred = logits[:, -1].argmax(-1)
            tgt = y[:, -1]
            correct += int((pred == tgt).sum())
            total += int(tgt.numel())
    acc = correct / max(1, total)
    return {
        "task": "associative_recall_2pair",
        "steps": steps,
        "final_train_loss": last_loss,
        "accuracy": acc,
        "chance": 1.0 / 8,
        "device": device,
        "params": model.param_report()["total"],
    }


def text_ce(path: Path, model_name: str, steps: int, device: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # char-level vocab
    chars = sorted(set(text)) or [" "]
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    if data.numel() < 16:
        return {"error": "text too short", "path": str(path)}

    V = len(chars)
    seq = 32
    if model_name == "aesc":
        model = SENTINEL(
            ModelConfig(
                d_model=64, d_k=32, d_v=32, n_skills=4, top_k=2, k_max=2,
                vocab_size=V, dendrite_window=3, max_seq_len=128, dropout=0.0,
            ),
            SafetyConfig(enable_audit=False),
        ).to(device)
        params = model.param_report()["total"]
    else:
        model = TinyTransformerLM(V, d_model=64, n_layers=2, n_heads=4, max_seq_len=128, dropout=0.0).to(device)
        params = model.param_count()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    t0 = time.time()
    last = None
    for step in range(1, steps + 1):
        if data.numel() <= seq + 1:
            i = 0
        else:
            i = torch.randint(0, data.numel() - seq - 1, (1,)).item()
        chunk = data[i : i + seq + 1]
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        if model_name == "aesc":
            loss = model(x, targets=y, lambda_err=0.01, lambda_bal=0.0).loss
        else:
            loss = model(x, targets=y).loss
        loss.backward()
        opt.step()
        last = float(loss.detach())
    elapsed = time.time() - t0
    return {
        "model": model_name,
        "path": str(path),
        "steps": steps,
        "final_ce": last,
        "params": params,
        "seconds": round(elapsed, 2),
        "device": device,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--recall-steps", type=int, default=150)
    ap.add_argument("--text-steps", type=int, default=100)
    ap.add_argument("--data", default="data/samples/experiment_pretrain.txt")
    ap.add_argument("--out", default="docs/audits/experiment_report_sample.json")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    report = {"associative_recall": associative_recall(args.recall_steps, device)}
    data = Path(args.data)
    if data.exists():
        report["aesc_text_ce"] = text_ce(data, "aesc", args.text_steps, device)
        if not args.skip_baseline:
            report["transformer_baseline_ce"] = text_ce(data, "transformer", args.text_steps, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
