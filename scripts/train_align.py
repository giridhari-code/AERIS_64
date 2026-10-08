#!/usr/bin/env python3
"""
Mini post-training for AERIS_64 (Anthropic-inspired, toy scale).

Stages (approximate mapping):
  1) SFT   — train on User/Assistant demos (+ constitution text as plain docs)
  2) Pref  — simple preference loss: increase likelihood of chosen vs rejected

This is NOT full Constitutional AI / RLAIF / PPO. It is a learning scaffold on a tiny model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from neurofield.checkpoint import load_checkpoint, save_checkpoint
from neurofield.config import ModelConfig, SafetyConfig, model_preset
from neurofield.model import NeuroField
from neurofield.tokenizer.char import CharTokenizer


def load_text_files(paths: list[str]) -> str:
    parts = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
            print(f"loaded {path}")
    return "\n".join(parts) + "\n"


def nll_on_text(model, encode, text: str, device, max_len: int = 128) -> torch.Tensor:
    ids = encode(text)
    if len(ids) < 3:
        return torch.tensor(0.0, device=device)
    ids = ids[: max_len + 1]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    out = model(x, targets=y, lambda_err=0.0, lambda_bal=0.0)
    return out.loss


def main() -> None:
    ap = argparse.ArgumentParser(description="AERIS mini align: SFT + simple preferences")
    ap.add_argument("--resume", default="docs/AERIS_64", help="base checkpoint")
    ap.add_argument("--sft", nargs="+", default=["data/align/sft_demos.txt", "data/align/constitution.txt"])
    ap.add_argument("--prefs", default="data/align/preferences.jsonl")
    ap.add_argument("--steps-sft", type=int, default=200)
    ap.add_argument("--steps-pref", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="docs/AERIS_64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preset", default="tiny")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    sft_text = load_text_files(args.sft)

    # tokenizer from resume vocab if possible else from sft text
    root = Path(args.resume)
    tok = CharTokenizer.from_text(sft_text)
    if (root / "vocab.json").is_file():
        try:
            raw = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
            if "stoi" in raw:
                tok = CharTokenizer(stoi=raw["stoi"], itos={int(k): v for k, v in raw.get("itos", {}).items()})
                print("using checkpoint vocab")
        except Exception as e:
            print("vocab load failed, using SFT text vocab", e)

    encode, decode = tok.encode, tok.decode
    vocab_size = tok.vocab_size

    preset_cfg = model_preset(args.preset)
    d_model = preset_cfg.d_model
    cfg = ModelConfig(
        d_model=d_model,
        d_k=max(8, d_model // 2),
        d_v=max(8, d_model // 2),
        n_skills=preset_cfg.n_skills,
        top_k=preset_cfg.top_k,
        k_max=preset_cfg.k_max,
        dendrite_window=preset_cfg.dendrite_window,
        vocab_size=vocab_size,
        max_seq_len=preset_cfg.max_seq_len,
        tie_embeddings=True,
        dropout=preset_cfg.dropout,
    )
    model = NeuroField(cfg, SafetyConfig(enable_audit=False)).to(device)

    if Path(args.resume).exists():
        try:
            ck = load_checkpoint(args.resume)
            from neurofield.checkpoint import remap_legacy_keys
            ck["model"], _renamed = remap_legacy_keys(ck["model"])
            # if vocab mismatch, still try partial load
            model.load_state_dict(ck["model"], strict=False)
            print(f"resumed weights from {args.resume}")
        except Exception as e:
            print("resume failed, scratch:", e)

    opt = AdamW(model.parameters(), lr=args.lr)

    # --- Stage 1: SFT-like ---
    print(f"SFT steps={args.steps_sft}")
    model.train()
    t0 = time.time()
    ids = torch.tensor(encode(sft_text), dtype=torch.long)
    seq = min(64, max(16, len(ids) // 4))
    for step in range(1, args.steps_sft + 1):
        if len(ids) < seq + 2:
            break
        hi = max(1, len(ids) - seq - 1)
        i = int(torch.randint(0, hi, (1,)).item())
        x = ids[i : i + seq].unsqueeze(0).to(device)
        y = ids[i + 1 : i + seq + 1].unsqueeze(0).to(device)
        loss = model(x, targets=y, lambda_err=0.02, lambda_bal=0.0).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"  sft {step} loss={float(loss):.3f}")
    print(f"SFT done in {time.time()-t0:.1f}s")

    # --- Stage 2: simple preference (chosen lower NLL than rejected) ---
    prefs = []
    pref_path = Path(args.prefs)
    if pref_path.is_file():
        for line in pref_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prefs.append(json.loads(line))
    print(f"Preference pairs={len(prefs)} steps={args.steps_pref}")
    for step in range(1, args.steps_pref + 1):
        if not prefs:
            break
        pair = prefs[(step - 1) % len(prefs)]
        prompt = pair["prompt"]
        chosen = f"User: {prompt}\nAssistant: {pair['chosen']}\n"
        rejected = f"User: {prompt}\nAssistant: {pair['rejected']}\n"
        loss_c = nll_on_text(model, encode, chosen, device)
        loss_r = nll_on_text(model, encode, rejected, device)
        # want loss_c < loss_r  →  minimize loss_c - loss_r (hinge-ish)
        loss = loss_c + F.relu(loss_c - loss_r + 0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"  pref {step} loss_c={float(loss_c):.3f} loss_r={float(loss_r):.3f}")

    config_dict = {
        "d_model": cfg.d_model,
        "d_k": cfg.d_k,
        "d_v": cfg.d_v,
        "n_skills": cfg.n_skills,
        "top_k": cfg.top_k,
        "k_max": cfg.k_max,
        "dendrite_window": cfg.dendrite_window,
        "vocab_size": cfg.vocab_size,
        "tie_embeddings": True,
        "tokenizer": "char",
    }
    out = save_checkpoint(
        args.out,
        model,
        config=config_dict,
        stoi=getattr(tok, "stoi", None),
        itos=getattr(tok, "itos", None),
        meta={
            "product": "AERIS_64",
            "stage": "align_sft_pref",
            "constitution": "data/align/constitution.txt",
            "note": "mini Anthropic-inspired post-train; not full CAI/RLAIF",
        },
    )
    (Path(out) / "tokenizer.json").write_text(
        json.dumps(tok.to_vocab_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved aligned checkpoint", out)


if __name__ == "__main__":
    main()
