#!/usr/bin/env python3
"""Company training entry: BPE or char, eval, safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sentinel.training.distributed import (
    init_distributed_if_needed,
    is_main,
    cleanup as dist_cleanup,
    world_info,
)
from torch.optim import AdamW

from sentinel.checkpoint import save_checkpoint
from sentinel.config import ModelConfig, SafetyConfig
from sentinel.eval.suite import run_eval_suite, save_eval_report
from sentinel.model import SENTINEL
from sentinel.tokenizer.bpe import BPETokenizer
from sentinel.tokenizer.char import CharTokenizer
from sentinel.utils.split import split_blocks
from sentinel.data.corpus import load_corpus
from sentinel.training.gpu import configure_accelerator, maybe_compile, autocast_ctx
from sentinel.training.paper_step import maybe_replay



def _forward_chunked(model, x, y, chunk_size, **kwargs):
    """Carry state across chunks of length C (reference form, sequential over chunks)."""
    if chunk_size <= 0 or x.size(1) <= chunk_size:
        return model(x, targets=y, **kwargs)
    B, T = x.shape
    state = None
    total_loss = None
    last_out = None
    n = 0
    for start in range(0, T, chunk_size):
        end = min(T, start + chunk_size)
        xc = x[:, start:end]
        yc = y[:, start:end] if y is not None else None
        out = model(xc, targets=yc, state=state, **kwargs)
        state = out.state
        if out.loss is not None:
            total_loss = out.loss if total_loss is None else total_loss + out.loss
            n += 1
        last_out = out
    if last_out is not None and total_loss is not None and n > 0:
        last_out.loss = total_loss / n
    return last_out


def main() -> None:
    init_distributed_if_needed()
    rank, world, local = world_info()
    ap = argparse.ArgumentParser(description="Train AERIS_64 (SENTINEL architecture)")
    ap.add_argument("--data", nargs="+", default=["data/company_corpus.txt", "data/code_corpus.txt", "data/india_multilang.txt", "data/skills_real.txt"],
                    help="Corpus: .txt .md .jsonl .json .csv .tsv .parquet .html (+.gz) or hf:dataset[:config:split]")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="bpe")
    ap.add_argument("--bpe-vocab", type=int, default=400)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=0, help="0 = from preset")
    ap.add_argument("--d-model", type=int, default=0, help="0 = use --preset")
    ap.add_argument("--preset", default="tiny", choices=["tiny", "small", "demo_1m", "medium", "large", "xl"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--warmup", type=int, default=50, help="LR warmup steps")
    ap.add_argument("--cosine", action="store_true", help="Cosine LR decay after warmup")
    ap.add_argument("--lr-schedule", choices=["constant", "cosine", "noam"], default="constant",
                    help="LR schedule: constant | cosine | noam (Transformer paper)")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="CE label smoothing (Transformer paper uses 0.1; 0=off)")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="Residual/embed dropout rate during train (paper 0.1)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--bf16", action="store_true", help="Force bf16 autocast (default: auto on CUDA)")
    ap.add_argument("--no-bf16", action="store_true", help="Disable bf16 even on CUDA")
    ap.add_argument("--compile", action="store_true", help="torch.compile model (PyTorch 2+)")
    ap.add_argument("--fast", action="store_true",
                    help="Speed mode: short seq, large batch, compile on CUDA (batch-parallel)")
    ap.add_argument("--chunk-parallel", type=int, default=0,
                    help="If >0, cut each seq into chunks of this len and stack on batch (more parallel sequences, shorter serial depth)")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="If >0, run forward in sequential chunks of this length carrying state (paper-style chunked train; C=1 is full seq)")
    ap.add_argument("--no-tf32", action="store_true", help="Disable TF32 on CUDA")
    ap.add_argument("--out", default="docs/sentinel_company")
    ap.add_argument("--device", default="auto", help="cpu | cuda | cuda:0 | auto")
    ap.add_argument("--resume", default="", help="Continue training from checkpoint folder")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction (de-duplicated, never trained on)")
    ap.add_argument("--product", default="", help="product/model name stored in meta (default: output folder name)")
    ap.add_argument("--identity-marker", default="", help="string expected in the 'who am I' reply (default: product)")
    args = ap.parse_args()

    # --fast: maximize batch parallelism, shorten sequential token loop
    if args.fast:
        if not args.seq_len:
            args.seq_len = 32
        if args.batch_size == 16:  # only if user left default
            args.batch_size = 64
        args.compile = True
        args.cosine = True
        if not getattr(args, "chunk_parallel", 0):
            args.chunk_parallel = 16
        if args.grad_accum < 1:
            args.grad_accum = 1
        print("[fast] seq_len=%d batch=%d compile=1 (sequence still serial; batch is parallel)" % (
            args.seq_len, args.batch_size))

    # Multi-format: .txt / .jsonl / .json → unified plain text (User:/Assistant: or prose)
    text, data_manifest = load_corpus(list(args.data))
    for m in data_manifest:
        print(f"loaded {m['path']} ({m['bytes']} B, {m['chars']} chars, format={m['format']})")
    if not text.strip():
        raise SystemExit("ERROR: no text loaded from --data (check paths and formats)")
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

    accel = configure_accelerator(
        args.device,
        bf16=False if args.no_bf16 else (True if args.bf16 else None),
        compile_model=args.compile,
        tf32=not args.no_tf32,
    )
    device = accel.device
    # torchrun: pin this process to its LOCAL_RANK GPU
    if world > 1 and torch.cuda.is_available() and str(device).startswith("cuda"):
        device = torch.device("cuda", int(local))
        torch.cuda.set_device(device)
    from sentinel.config import model_preset
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
        dropout=float(args.dropout) if args.dropout is not None else preset_cfg.dropout,
    )
    print(f"preset={args.preset} d_model={d_model} seq_len={seq_len} max_seq_len={cfg.max_seq_len}")
    model = SENTINEL(cfg, SafetyConfig(enable_audit=False, max_memory_norm=40.0)).to(device)
    raw_model = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,  # True only if ablations leave params unused,
        )
        if is_main():
            print(f"[distributed] DDP on {world} ranks")
    model = maybe_compile(model, accel.use_compile)
    if args.resume:
        from sentinel.checkpoint import load_checkpoint
        try:
            ck = load_checkpoint(args.resume)
            from sentinel.checkpoint import remap_legacy_keys
            ck["model"], _renamed = remap_legacy_keys(ck["model"])
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            print(f"resumed from {args.resume} missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"resume failed ({e}); training from scratch")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Keep corpus on device for faster slice assembly (batch axis = parallel sequences)
    ids = ids.to(device)
    n_tok = int(ids.numel())
    if n_tok < 8:
        raise SystemExit(
            f"ERROR: only {n_tok} tokens in corpus after tokenize — "
            "refusing to train. Provide a longer non-empty --data file."
        )
    # Short corpus: shrink seq so windows fit (avoids CUDA index OOB)
    if n_tok < args.seq_len + 2:
        new_seq = max(8, n_tok - 2)
        if is_main():
            print(
                f"WARN: only {n_tok} tokens; seq_len {args.seq_len} → {new_seq} "
                f"(use longer data for real training)"
            )
        args.seq_len = new_seq
    V = int(cfg.vocab_size)
    ids = ids.clamp(0, V - 1)

    def batch():
        bs, seq = args.batch_size, args.seq_len
        # hi exclusive upper for start index so idx+1 stays in range
        hi = max(1, n_tok - seq - 1)
        ix = torch.randint(0, hi, (bs,), device=device)
        offsets = torch.arange(seq, device=device).unsqueeze(0)
        idx = ix.unsqueeze(1) + offsets
        # hard safety for tiny corpora
        idx = idx.clamp(0, max(0, n_tok - 2))
        x = ids[idx].clamp(0, V - 1)
        y = ids[(idx + 1).clamp(0, n_tok - 1)].clamp(0, V - 1)
        # Chunk-parallel: (B, S) -> (B*n_chunks, chunk) so GPU sees more parallel rows,
        # each with a short serial token loop (architecture throughput knob).
        cp = int(getattr(args, "chunk_parallel", 0) or 0)
        if cp > 0 and seq >= cp * 2:
            n_chunks = seq // cp
            use = n_chunks * cp
            x = x[:, :use].reshape(bs * n_chunks, cp)
            y = y[:, :use].reshape(bs * n_chunks, cp)
        return x, y

    use_bf16 = accel.use_bf16
    # Paper v3: surprise-weighted replay into W_s every `replay_every` steps
    replay_every = 50
    replay_eta = 0.02
    print(
        f"Training steps={args.steps} device={device} accum={args.grad_accum} world={world} "
        f"schedule={getattr(args, "lr_schedule", "constant")} cosine={args.cosine} smooth={getattr(args, "label_smoothing", 0)} drop={getattr(args, "dropout", 0)} bf16={use_bf16} compile={accel.use_compile} "
        f"tf32={accel.tf32} replay_every={replay_every}"
    )

    t0 = time.time()
    model.train()
    last_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        # LR schedule: linear warmup + optional cosine
        import math
        schedule = getattr(args, "lr_schedule", "constant")
        if args.cosine:
            schedule = "cosine"
        if schedule == "noam":
            # Attention Is All You Need §5.3: d^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
            # args.lr acts as an extra scale (default ~1.0 for pure formula; often use 1.0)
            warm = max(1, args.warmup if args.warmup > 0 else 4000)
            step_f = float(step)
            inv_sqrt_d = (float(d_model) ** -0.5)
            lr_now = inv_sqrt_d * min(step_f ** -0.5, step_f * (warm ** -1.5))
            lr_now = lr_now * float(args.lr)
        elif schedule == "cosine":
            if args.warmup > 0 and step <= args.warmup:
                lr_scale = step / max(1, args.warmup)
            else:
                progress = (step - args.warmup) / max(1, args.steps - args.warmup)
                lr_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            lr_now = args.lr * lr_scale
        else:
            # constant + optional linear warmup
            if args.warmup > 0 and step <= args.warmup:
                lr_scale = step / max(1, args.warmup)
            else:
                lr_scale = 1.0
            lr_now = args.lr * lr_scale
        for g in opt.param_groups:
            g["lr"] = lr_now

        x, y = batch()
        with autocast_ctx(accel):
            fw_kwargs = dict(
                lambda_err=0.03, lambda_bal=0.02,
                label_smoothing=float(getattr(args, "label_smoothing", 0.0) or 0.0),
                collect_stats=(step % 100 == 0 or step == 1),
            )
            if getattr(args, "chunk_size", 0) and args.chunk_size > 0:
                out = _forward_chunked(model, x, y, args.chunk_size, **fw_kwargs)
            else:
                out = model(x, targets=y, **fw_kwargs)
        loss = out.loss
        (loss / max(1, args.grad_accum)).backward()
        if step % max(1, args.grad_accum) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Paper Appendix A: W_s ← (1-η)W_s + η Σ_b (S^b / Σ S) M^b
        # Also honour metacognition want_replay when the control layer requests consolidation.
        want_replay = bool((out.state or {}).get("want_replay")) or bool(
            (out.metrics or {}).get("want_replay")
        )
        if out.state is not None and "M" in out.state and (
            step % replay_every == 0 or want_replay
        ):
            S = out.metrics.get("_seq_surprise") if out.metrics else None
            if S is None or not torch.is_tensor(S):
                S = torch.ones(x.size(0), device=device)
            model.replay_slow(out.state["M"], S, replay_eta)
            if want_replay and hasattr(model, "meta") and hasattr(model.meta, "acknowledge_replay"):
                model.meta.acknowledge_replay()

        last_loss = float(loss.item())
        if step % 100 == 0 or step == 1:
            met = out.metrics or {}
            gate = met.get("gate_mean", 0.0)
            slow_f = met.get("slow_filled", 0.0)
            k_d = met.get("k_depth", 0.0)
            elapsed = max(1e-6, time.time() - t0)
            tokens_done = step * args.batch_size * args.seq_len
            tok_s = tokens_done / elapsed
            print(
                f"  step {step:5d} | loss={last_loss:.3f} | "
                f"lr={lr_now:.2e} | gate={gate:.3f} | slow={slow_f:.0f} | k={k_d:.0f} | "
                f"{tok_s:.0f} tok/s | seq={args.seq_len} bs={args.batch_size}"
            )
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

    # Always save the unwrapped module (compile wraps as OptimizedModule with
    # _orig_mod.* keys — those break server load looking for embed.weight).
    to_save = raw_model
    if not is_main():
        dist_cleanup()
        return
    out = save_checkpoint(
        args.out,
        to_save,
        config=config_dict,
        stoi=stoi,
        itos=itos,
        meta={
            "uses_specials": bool(getattr(tok, "uses_specials", False)),

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
        to_save, encode, decode, text=train_text, val_text=val_text, device=str(device),
        identity_marker=args.identity_marker or product,
    )
    report["split"] = split_stats
    save_eval_report(report, Path(out) / "eval_report.json")
    print("held-out val_nll", report.get("val_nll"), "| train_nll", report.get("train_nll"),
          "| identity_hit", report.get("identity_hit"), "| warnings", report.get("warnings"))
    print("Saved", out)
    dist_cleanup()


if __name__ == "__main__":
    main()
