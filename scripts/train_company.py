#!/usr/bin/env python3
"""Company training entry: BPE or char, eval, safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from neurofield.checkpoint import save_checkpoint
from neurofield.config import ModelConfig, SafetyConfig
from neurofield.eval.suite import run_eval_suite, save_eval_report
from neurofield.model import NeuroField
from neurofield.tokenizer.bpe import BPETokenizer
from neurofield.tokenizer.char import CharTokenizer
from neurofield.utils.split import split_blocks


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AERIS_64 (NeuroField architecture)")
    ap.add_argument("--data", nargs="+", default=["data/company_corpus.txt", "data/code_corpus.txt", "data/india_multilang.txt", "data/skills_real.txt"])
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="bpe")
    ap.add_argument("--bpe-vocab", type=int, default=400)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=0, help="0 = from preset")
    ap.add_argument("--d-model", type=int, default=0, help="0 = use --preset")
    ap.add_argument("--preset", default="tiny", choices=["tiny", "small", "medium", "large", "xl"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--warmup", type=int, default=50, help="LR warmup steps")
    ap.add_argument("--cosine", action="store_true", help="Cosine LR decay after warmup")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--bf16", action="store_true", help="Autocast bfloat16 if CUDA")
    ap.add_argument("--out", default="docs/neurofield_company")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", default="", help="Continue training from checkpoint folder")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction (de-duplicated, never trained on)")
    ap.add_argument("--product", default="", help="product/model name stored in meta (default: output folder name)")
    ap.add_argument("--identity-marker", default="", help="string expected in the 'who am I' reply (default: product)")
    args = ap.parse_args()

    parts = []
    for path in args.data:
        p = Path(path)
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
            print(f"loaded {p} ({p.stat().st_size} B)")
    text = "\n".join(parts)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    text = "\n".join(lines) + "\n"
    # Hold out BEFORE tokenizer training / training so nothing leaks into val.
    train_text, val_text, split_stats = split_blocks(text, args.val_frac)
    print(f"split: {split_stats} (duplicates removed before split)")
    text = train_text

    if args.tokenizer == "bpe":
        try:
            tok = BPETokenizer.train(text, vocab_size=args.bpe_vocab)
            encode = tok.encode
            decode = tok.decode
            vocab_size = tok.vocab_size
            print(f"BPE vocab_size={vocab_size} merges={len(tok.merges)}")
        except Exception as e:
            print(f"BPE failed ({e}); falling back to char")
            args.tokenizer = "char"
            tok = CharTokenizer.from_text(text)
            encode = tok.encode
            decode = tok.decode
            vocab_size = tok.vocab_size
            print(f"Char vocab_size={vocab_size}")
    else:
        tok = CharTokenizer.from_text(text)
        encode = tok.encode
        decode = tok.decode
        vocab_size = tok.vocab_size
        print(f"Char vocab_size={vocab_size}")

    ids = torch.tensor(encode(text), dtype=torch.long)
    print(f"tokens={len(ids)}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    from neurofield.config import model_preset
    preset_cfg = model_preset(args.preset)
    d_model = args.d_model if args.d_model and args.d_model > 0 else preset_cfg.d_model
    seq_len = args.seq_len if getattr(args, "seq_len", 0) and args.seq_len > 0 else max(32, min(256, preset_cfg.max_seq_len // 2))
    args.seq_len = seq_len  # for batch()
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
    print(f"preset={args.preset} d_model={d_model} seq_len={seq_len} max_seq_len={cfg.max_seq_len}")
    model = NeuroField(cfg, SafetyConfig(enable_audit=False, max_memory_norm=40.0)).to(device)
    if args.resume:
        from neurofield.checkpoint import load_checkpoint
        try:
            ck = load_checkpoint(args.resume)
            from neurofield.checkpoint import remap_legacy_keys
            ck["model"], _renamed = remap_legacy_keys(ck["model"])
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            print(f"resumed from {args.resume} missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"resume failed ({e}); training from scratch")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch():
        bs, seq = args.batch_size, args.seq_len
        hi = max(1, len(ids) - seq - 1)
        ix = torch.randint(0, hi, (bs,))
        x = torch.stack([ids[i : i + seq] for i in ix]).to(device)
        y = torch.stack([ids[i + 1 : i + seq + 1] for i in ix]).to(device)
        return x, y

    use_bf16 = bool(args.bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Training steps={args.steps} device={device} accum={args.grad_accum} cosine={args.cosine} bf16={use_bf16}")
    t0 = time.time()
    model.train()
    last_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        # LR schedule: linear warmup + optional cosine
        if args.warmup > 0 and step <= args.warmup:
            lr_scale = step / max(1, args.warmup)
        elif args.cosine:
            import math
            progress = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        else:
            lr_scale = 1.0
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_scale

        x, y = batch()
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, targets=y, lambda_err=0.03, lambda_bal=0.005).loss
        else:
            loss = model(x, targets=y, lambda_err=0.03, lambda_bal=0.005).loss
        (loss / max(1, args.grad_accum)).backward()
        if step % max(1, args.grad_accum) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        last_loss = float(loss.item())
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d} | loss={last_loss:.3f} | lr={args.lr * lr_scale:.2e}")
    if any(p.grad is not None for p in model.parameters() if p.requires_grad):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    print(f"Done in {time.time() - t0:.1f}s final_loss={last_loss:.3f}")

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
        "tokenizer": args.tokenizer,
    }
    stoi = getattr(tok, "stoi", None)
    itos = getattr(tok, "itos", None)
    if args.tokenizer == "char":
        meta_tok = tok.to_vocab_dict()
    else:
        meta_tok = tok.to_vocab_dict()
        stoi, itos = None, None

    out = save_checkpoint(
        args.out,
        model,
        config=config_dict,
        stoi=stoi,
        itos=itos,
        meta={
            "steps": args.steps,
            "final_loss": last_loss,
            "tokenizer": args.tokenizer,
            "data": args.data,
            "model_id": args.product or Path(args.out).name,
            "product": args.product or Path(args.out).name,
            "val_fraction": args.val_frac,
            "split": split_stats,
        },
    )
    # save tokenizer sidecar
    if args.tokenizer == "bpe":
        tok.save(Path(out) / "tokenizer.json")
    else:
        (Path(out) / "tokenizer.json").write_text(json.dumps(meta_tok, ensure_ascii=False, indent=2), encoding="utf-8")

    product = args.product or Path(args.out).name
    report = run_eval_suite(
        model, encode, decode, text=train_text, val_text=val_text, device=str(device),
        identity_marker=args.identity_marker or product,
    )
    report["split"] = split_stats
    save_eval_report(report, Path(out) / "eval_report.json")
    print("held-out val_nll", report.get("val_nll"), "| train_nll", report.get("train_nll"),
          "| identity_hit", report.get("identity_hit"), "| warnings", report.get("warnings"))
    print("Saved", out)


if __name__ == "__main__":
    main()
