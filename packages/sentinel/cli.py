"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from sentinel.config import load_config, SENTINELConfig
from sentinel.data.dataset import build_dataloader
from sentinel.model import SENTINEL
from sentinel.training import Trainer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train SENTINEL v2")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--train-data", type=str, default=None,
                        help="token .bin/.npy, directory of those, or glob")
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="auto-pick train.bin + val.bin (or train*.bin) from this folder")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from <output-dir>/checkpoint_latest.pt")
    parser.add_argument("--tokenizer-json", type=str, default=None, help="copied into the exported checkpoint folders")
    parser.add_argument("--token-dtype", type=str, default="uint16",
                        choices=["uint16", "int32", "auto"],
                        help="dtype of the .bin token files (auto = infer from sidecar/name)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--streaming", action="store_true",
                        help="infinite random windows (better for multi-GB corpora)")
    parser.add_argument("--stride", type=int, default=1,
                        help="map-style window stride (seq_len = non-overlapping packs)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.data_dir:
        d = Path(args.data_dir)
        if args.train_data is None:
            for name in ("train.bin", "train.npy"):
                if (d / name).exists():
                    cfg.data.train_path = str(d / name)
                    break
            else:
                # fall back to whole directory (multi-file)
                cfg.data.train_path = str(d)
        if args.val_data is None:
            for name in ("val.bin", "valid.bin", "val.npy"):
                if (d / name).exists():
                    cfg.data.val_path = str(d / name)
                    break
    if args.train_data:
        cfg.data.train_path = args.train_data
    if args.val_data:
        cfg.data.val_path = args.val_data
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.device:
        cfg.training.device = args.device
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.training.seq_len = args.seq_len
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.grad_accum is not None:
        cfg.training.grad_accum_steps = args.grad_accum
    if args.precision is not None:
        cfg.training.precision = args.precision
    if args.seed is not None:
        cfg.training.seed = args.seed

    setup_logging(cfg.logging.level)
    try:
        import safetensors  # noqa: F401
    except ImportError:
        logging.warning(
            "safetensors not installed — checkpoint export may fall back to model.pt. "
            "pip install safetensors (required for tied-embedding exports)."
        )
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        logging.warning(
            "tokenizers not installed — prepare_data / production encode need: pip install tokenizers"
        )
    from sentinel.training.distributed import cleanup, get_world_size, init_distributed_if_needed
    from sentinel.data.dataset import estimate_tokens_per_step

    init_distributed_if_needed()  # torchrun => data-parallel, plain `python` => unchanged
    torch.manual_seed(cfg.training.seed)

    if cfg.data.train_path is None:
        logging.error(
            "No training data provided. Set data.train_path in config, --train-data, or --data-dir. "
            "Raw .txt is not accepted here — run scripts/data/prepare_data.py first to create train.bin."
        )
        sys.exit(1)
    # Early, clear error if path points at a folder without token bins
    from pathlib import Path as _P
    _tp = _P(cfg.data.train_path)
    if _tp.is_dir() and not any(_tp.glob("*.bin")) and not any(_tp.glob("*.npy")):
        logging.error(
            "No .bin/.npy under %s. Tokenize first: "
            "python scripts/data/prepare_data.py --input 'your_data/*.txt' --out-dir %s --vocab-size 8000",
            _tp, _tp,
        )
        sys.exit(1)

    if args.token_dtype == "uint16" and cfg.model.vocab_size > 65535:
        logging.error("vocab_size %d does not fit uint16; use --token-dtype int32", cfg.model.vocab_size)
        sys.exit(1)

    model = SENTINEL(cfg.model, cfg.safety)
    report = model.param_report()
    logging.info("Parameter report: %s", report)

    tok_step = estimate_tokens_per_step(
        cfg.training.batch_size * max(1, int(getattr(cfg.training, "grad_accum_steps", 1) or 1)),
        cfg.training.seq_len,
        get_world_size(),
    )
    logging.info(
        "Train plan: steps=%d seq_len=%d batch=%d accum=%d precision=%s ~tokens/step=%s",
        cfg.training.max_steps,
        cfg.training.seq_len,
        cfg.training.batch_size,
        getattr(cfg.training, "grad_accum_steps", 1),
        cfg.training.precision,
        f"{tok_step:,}",
    )

    # Estimate corpus size; auto-stream on tiny bins (map-style can explode RAM)
    use_streaming = bool(args.streaming)
    n_train_tok = 0
    try:
        from sentinel.data.dataset import count_tokens
        n_train_tok = count_tokens(cfg.data.train_path, dtype=args.token_dtype)
    except Exception as e:
        logging.debug("token count probe failed: %s", e)
    if n_train_tok > 0:
        logging.info("Train corpus: ~%s tokens", f"{n_train_tok:,}")
        if n_train_tok < 50_000:
            logging.warning(
                "Small corpus (~%s tokens). Expect high CE and weak generations until you "
                "add much more text and train longer. Pipeline success ≠ an intelligent model.",
                f"{n_train_tok:,}",
            )
            if not use_streaming:
                use_streaming = True
                logging.info("Auto-enabled --streaming for small corpus (safer on CPU/low RAM).")
        if n_train_tok < 5_000_000 and cfg.training.max_steps < 2000:
            logging.warning(
                "max_steps=%d is low for ~%s tokens if you want usable quality; "
                "increase steps and data for product models. 1B-class needs multi-GPU + billions of tokens (not Colab free).",
                cfg.training.max_steps,
                f"{n_train_tok:,}",
            )

    train_loader = build_dataloader(
        cfg.data.train_path,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        num_workers=args.num_workers,
        dtype=args.token_dtype,
        streaming=use_streaming,
        stride=max(1, args.stride),
        shuffle_train=True,
    )
    val_loader = build_dataloader(
        cfg.data.val_path,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        num_workers=0,
        dtype=args.token_dtype,
        streaming=False,
        stride=max(1, cfg.training.seq_len),  # non-overlapping val packs
        shuffle_train=False,
        drop_last=False,
    )
    if val_loader is None:
        logging.warning("No val_path: training WITHOUT held-out validation (no best-checkpoint, no ppl).")

    extra = {"tokenizer.json": args.tokenizer_json} if args.tokenizer_json else None
    trainer = Trainer(model, cfg, train_loader, val_loader, extra_files=extra)
    if args.resume:
        logging.info("resume: %s", "found checkpoint" if trainer.resume() else "no checkpoint, starting fresh")
    trainer.train()
    cleanup()


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SENTINEL checkpoint on a tokenized .bin/.npy")
    parser.add_argument("--checkpoint", type=str, required=True, help="safetensors folder or checkpoint_*.pt")
    parser.add_argument("--data", type=str, required=True, help="HELD-OUT token file")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--token-dtype", type=str, default="uint16", choices=["uint16", "int32"])
    parser.add_argument("--max-batches", type=int, default=200)
    args = parser.parse_args(argv)

    setup_logging()
    from sentinel.checkpoint.resolve import load_model_from_checkpoint

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, _tok, cfg, _meta = load_model_from_checkpoint(args.checkpoint, args.config, device=device)

    loader = build_dataloader(
        args.data,
        seq_len=args.seq_len or cfg.training.seq_len,
        batch_size=8,
        dtype=args.token_dtype,
    )
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if n >= args.max_batches:
                break
            x = batch["input_ids"].to(device)
            y = batch["targets"].to(device)
            out = model(x, targets=y)
            total += out.metrics["ce"]  # pure cross-entropy
            n += 1
    ce = total / max(1, n)
    print(f"Held-out cross-entropy: {ce:.4f} nats/token  (ppl {torch.exp(torch.tensor(min(ce, 20.0))).item():.2f}) over {n} batches")


def pipeline_main(argv: list[str] | None = None) -> None:
    """Multi-stage company pipeline: pretrain → … → dpo / grpo / rlhf / opd."""
    from sentinel.pipeline.registry import list_stages
    from sentinel.pipeline.runner import DEFAULT_ORDER, run_pipeline

    parser = argparse.ArgumentParser(
        description="SENTINEL multi-stage training pipeline (company standard)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=f"Comma-separated stages. Default full order. Known: {', '.join(list_stages() or DEFAULT_ORDER)}",
    )
    parser.add_argument("--config", type=str, default=None, help="Base model/training YAML")
    parser.add_argument(
        "--pipeline-config",
        type=str,
        default=None,
        help="Pipeline YAML (stages + per-stage overrides). See configs/pipeline/default.yaml",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Folder with train.bin / val.bin")
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--sft-data", type=str, default=None, help="SFT jsonl or txt")
    parser.add_argument("--prefs", type=str, default=None, help="Preference jsonl (prompt/chosen/rejected)")
    parser.add_argument("--reasoning-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/pipeline")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint.pt to warm-start")
    parser.add_argument("--tokenizer-json", type=str, default=None,
                        help="tokenizer.json from prepare_data.py (required for production SFT/prefs)")
    parser.add_argument("--list-stages", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    # ensure stages registered
    import sentinel.pipeline.stages  # noqa: F401

    if args.list_stages:
        print("Available stages:")
        for s in list_stages():
            print(f"  - {s}")
        print("Default order:", " → ".join(DEFAULT_ORDER))
        return

    train_bin = args.train_data
    val_bin = args.val_data
    if args.data_dir:
        d = Path(args.data_dir)
        if train_bin is None:
            for name in ("train.bin", "train.npy"):
                if (d / name).exists():
                    train_bin = str(d / name)
                    break
            else:
                train_bin = str(d)
        if val_bin is None:
            for name in ("val.bin", "valid.bin", "val.npy"):
                if (d / name).exists():
                    val_bin = str(d / name)
                    break


    tok = args.tokenizer_json
    if tok is None and args.data_dir:
        cand = Path(args.data_dir) / "tokenizer.json"
        if cand.is_file():
            tok = str(cand)
    args.tokenizer_json = tok

    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    run_pipeline(
        stages=stages,
        config_path=args.config,
        pipeline_yaml=args.pipeline_config,
        output_dir=args.output_dir,
        train_bin=train_bin,
        val_bin=val_bin,
        sft_path=args.sft_data,
        prefs_path=args.prefs,
        reasoning_path=args.reasoning_data,
        device=args.device,
        resume_checkpoint=args.resume,
        tokenizer_json=args.tokenizer_json,
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatcher: train | eval | pipeline."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: python -m sentinel.cli <train|eval|pipeline> [options]")
        print("  train     — single-stage LM training (pretrain-style)")
        print("  eval      — held-out CE evaluation")
        print("  pipeline  — multi-stage: pretrain,sft,dpo,grpo,rlhf_ppo,...")
        print("  pipeline --list-stages")
        sys.exit(0)
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        train_main(rest)
    elif cmd == "eval":
        eval_main(rest)
    elif cmd == "pipeline":
        pipeline_main(rest)
    else:
        # backward compat: flags without subcommand → train
        train_main(argv)


if __name__ == "__main__":
    main()
