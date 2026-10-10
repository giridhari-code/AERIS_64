#!/usr/bin/env python3
"""
Knowledge distillation: student mimics teacher soft labels (or teacher generations).

Modes:
  --mode soft     KL(student || teacher) on next-token distributions (same architecture family)
  --mode sft      Generate with teacher offline first, then SFT student on text (use train_finetune)

For different-size NeuroField students, soft mode requires same vocab and compatible heads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill teacher → student")
    ap.add_argument("--teacher", required=True, help="teacher checkpoint folder")
    ap.add_argument("--student", default="", help="student checkpoint to resume (optional)")
    ap.add_argument("--data", required=True, nargs="+", help="text files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on distill KL vs CE")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preset", default="small")
    args = ap.parse_args()

    from neurofield.checkpoint.infer import load_model_from_checkpoint
    from neurofield.checkpoint.io import save_checkpoint
    from neurofield.config import ModelConfig
    # train helpers from company if available
    try:
        from neurofield.model import NeuroField
    except ImportError:
        pass

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    teacher, tok, tcfg, _ = load_model_from_checkpoint(args.teacher, None, device=str(device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if args.student:
        student, _, scfg, _ = load_model_from_checkpoint(args.student, None, device=str(device))
    else:
        # new student from teacher config, optionally smaller via preset later
        student, _, scfg, _ = load_model_from_checkpoint(args.teacher, None, device=str(device))
        # re-init student weights lightly: keep architecture, fresh train
        student.train()

    student.train()
    student.to(device)
    teacher.to(device)

    text = ""
    for fp in args.data:
        text += Path(fp).read_text(encoding="utf-8", errors="ignore") + "\n"
    ids = tok.encode(text)
    if len(ids) < args.seq_len + 2:
        ids = ids * ((args.seq_len + 2) // max(1, len(ids)) + 1)
    ids_t = torch.tensor(ids, dtype=torch.long)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    T = args.temperature
    alpha = args.alpha

    print(f"distill steps={args.steps} alpha={alpha} T={T} device={device}")
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(args.batch_size):
            i = torch.randint(0, len(ids_t) - args.seq_len - 1, (1,)).item()
            batch.append(ids_t[i : i + args.seq_len + 1])
        batch = torch.stack(batch).to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        with torch.no_grad():
            t_out = teacher(x)
            t_logits = t_out.logits
        s_out = student(x, targets=y)
        s_logits = s_out.logits

        # CE on hard labels
        ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), y.reshape(-1), ignore_index=-100)
        # soft KL
        log_s = F.log_softmax(s_logits / T, dim=-1)
        soft_t = F.softmax(t_logits / T, dim=-1)
        kl = F.kl_div(log_s, soft_t, reduction="batchmean") * (T * T)
        loss = alpha * kl + (1 - alpha) * ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step {step} loss={loss.item():.4f} ce={ce.item():.4f} kl={kl.item():.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = getattr(student, "cfg", None)
    config = cfg.__dict__ if cfg is not None and hasattr(cfg, "__dict__") else {}
    save_checkpoint(
        out,
        student,
        config=config,
        stoi=getattr(tok, "stoi", None),
        itos=getattr(tok, "itos", None),
        meta={"format": "neurofield-safetensors-v1", "distill_from": args.teacher, "steps": args.steps},
    )
    print(f"saved student → {out}")


if __name__ == "__main__":
    main()
