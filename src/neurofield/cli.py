"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from neurofield.config import load_config, NeuroFieldConfig
from neurofield.data import build_dataloader
from neurofield.model import NeuroField
from neurofield.training import Trainer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train NeuroField v2")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from <output-dir>/checkpoint_latest.pt")
    parser.add_argument("--tokenizer-json", type=str, default=None, help="copied into the exported checkpoint folders")
    parser.add_argument("--token-dtype", type=str, default="uint16", choices=["uint16", "int32"],
                        help="dtype of the .bin token files (uint16 holds vocab <= 65535)")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
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
    if args.seed is not None:
        cfg.training.seed = args.seed

    setup_logging(cfg.logging.level)
    torch.manual_seed(cfg.training.seed)

    if cfg.data.train_path is None:
        logging.error(
            "No training data provided. Set data.train_path in config or use --train-data."
        )
        sys.exit(1)

    if args.token_dtype == "uint16" and cfg.model.vocab_size > 65535:
        logging.error("vocab_size %d does not fit uint16; use --token-dtype int32", cfg.model.vocab_size)
        sys.exit(1)

    model = NeuroField(cfg.model, cfg.safety)
    report = model.param_report()
    logging.info("Parameter report: %s", report)

    train_loader = build_dataloader(
        cfg.data.train_path,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        num_workers=args.num_workers,
        dtype=args.token_dtype,
    )
    val_loader = build_dataloader(
        cfg.data.val_path,
        seq_len=cfg.training.seq_len,
        batch_size=cfg.training.batch_size,
        num_workers=0,
        dtype=args.token_dtype,
    )
    if val_loader is None:
        logging.warning("No val_path: training WITHOUT held-out validation (no best-checkpoint, no ppl).")

    extra = {"tokenizer.json": args.tokenizer_json} if args.tokenizer_json else None
    trainer = Trainer(model, cfg, train_loader, val_loader, extra_files=extra)
    if args.resume:
        logging.info("resume: %s", "found checkpoint" if trainer.resume() else "no checkpoint, starting fresh")
    trainer.train()


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a NeuroField checkpoint on a tokenized .bin/.npy")
    parser.add_argument("--checkpoint", type=str, required=True, help="safetensors folder or checkpoint_*.pt")
    parser.add_argument("--data", type=str, required=True, help="HELD-OUT token file")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--token-dtype", type=str, default="uint16", choices=["uint16", "int32"])
    parser.add_argument("--max-batches", type=int, default=200)
    args = parser.parse_args(argv)

    setup_logging()
    from neurofield.checkpoint.resolve import load_model_from_checkpoint

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


if __name__ == "__main__":
    train_main()
