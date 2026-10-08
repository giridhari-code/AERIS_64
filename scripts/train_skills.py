#!/usr/bin/env python3
"""Train NeuroField on real skills dataset (character-level)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/skills_real.txt")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--out", default="docs/neurofield_skills.pt")
    p.add_argument("--d-model", type=int, default=64)
    args = p.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"chars={len(data)} vocab={len(chars)}")

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

    def batch(bs: int = 16, seq: int = 48):
        ix = torch.randint(0, len(data) - seq - 1, (bs,))
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    from neurofield.checkpoint import save_checkpoint
    cfg_dict = {
        "d_model": cfg.d_model,
        "d_k": cfg.d_k,
        "d_v": cfg.d_v,
        "n_skills": cfg.n_skills,
        "top_k": cfg.top_k,
        "k_max": cfg.k_max,
        "dendrite_window": cfg.dendrite_window,
        "vocab_size": cfg.vocab_size,
        "tie_embeddings": True,
    }
    out_path = save_checkpoint(
        args.out,
        model,
        config=cfg_dict,
        stoi=stoi,
        itos=itos,
        meta={"source": "skills_real.txt", "steps": args.steps},
    )
    print(f"Saved safetensors checkpoint folder: {out_path}")
    print(f"  - {out_path}/model.safetensors")
    print(f"  - {out_path}/config.json")
    print(f"  - {out_path}/vocab.json")

    # quick samples
    def gen(prompt: str, n: int = 80, temp: float = 0.35) -> str:
        model.eval()
        ids = torch.tensor([[stoi.get(c, 0) for c in prompt]])
        out_chars = list(prompt)
        state = None
        cur = ids
        with torch.no_grad():
            for _ in range(n):
                o = model(cur, state=state)
                state = o.state
                logits = o.logits[0, -1] / temp
                nid = torch.multinomial(torch.softmax(logits, -1), 1).item()
                out_chars.append(itos.get(nid, "?"))
                cur = torch.tensor([[nid]])
        return "".join(out_chars)

    print("\n=== samples ===")
    for pr in ["Self-learning", "Emotional Intelligence", "Never depend", "Control your"]:
        print(f"IN : {pr}")
        print(f"OUT: {gen(pr)[:150].replace(chr(10), ' | ')}")
        print()


if __name__ == "__main__":
    main()
