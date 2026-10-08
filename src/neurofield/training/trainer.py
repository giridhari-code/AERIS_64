"""Training loop for NeuroField (1B-ready).

Adds over the original loop:
  * gradient accumulation (training.grad_accum_steps)
  * bf16 autocast (training.precision = "bf16")
  * resume from checkpoint_latest.pt
  * atomic checkpoint writes + a safetensors export of the final / best model
  * training.truncate_write_window is now actually applied to the model
    (before, the model silently used a hardcoded 32)
  * slow-memory replay uses the real per-sequence surprise
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from neurofield.checkpoint.io import _torch_load, save_checkpoint
from neurofield.config import NeuroFieldConfig, save_config
from neurofield.model import NeuroField

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _decay_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """No weight decay on 1-D params (biases, norms, gates) or the tied embedding."""
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (no_decay if (p.ndim < 2 or name.startswith(("embed.", "slow_mem.M_s"))) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Trainer:
    def __init__(
        self,
        model: NeuroField,
        cfg: NeuroFieldConfig,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        extra_files: Optional[dict[str, str]] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.extra_files = extra_files or {}  # name -> source path copied next to exports

        tc = cfg.training
        self.device = torch.device(tc.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # honour the config instead of the model's hardcoded default
        self.model._truncate_window = max(1, int(tc.truncate_write_window))

        if tc.precision not in ("fp32", "bf16"):
            raise ValueError(f"training.precision must be fp32|bf16, got {tc.precision!r}")
        self.use_bf16 = tc.precision == "bf16"
        self.accum = max(1, int(tc.grad_accum_steps))

        self.optimizer = AdamW(
            _decay_groups(self.model, tc.weight_decay),
            lr=tc.lr,
            betas=(0.9, 0.95),
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=tc.warmup_steps,
            num_training_steps=tc.max_steps,
        )

        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.output_dir / "config.yaml")

        self.global_step = 0
        self.best_val = float("inf")

    # ------------------------------------------------------------------ io
    def _atomic_save(self, obj: Any, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _save_checkpoint(self, tag: str = "latest") -> None:
        path = self.output_dir / f"checkpoint_{tag}.pt"
        self._atomic_save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.global_step,
                "best_val": self.best_val,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint %s", path)

    def resume(self, path: Optional[str | Path] = None) -> bool:
        """Load optimizer/scheduler/model state. Returns True if a checkpoint was found."""
        path = Path(path) if path else self.output_dir / "checkpoint_latest.pt"
        if not path.is_file():
            return False
        # trusted: our own file. weights_only path still works for tensors + plain dicts.
        ckpt = _torch_load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = int(ckpt.get("step", 0))
        self.best_val = float(ckpt.get("best_val", float("inf")))
        logger.info("Resumed from %s at step %d", path, self.global_step)
        return True

    def export_safetensors(self, name: str = "final") -> Path:
        """Weights-only export (no pickle) + config.json + meta.json (+ tokenizer files)."""
        out = save_checkpoint(
            self.output_dir / name,
            self.model,
            config=self.cfg.model.__dict__.copy(),
            meta={
                "model_id": self.output_dir.name,
                "step": self.global_step,
                "params": self.model.param_report()["total"],
                "precision": self.cfg.training.precision,
                "best_val_loss": None if math.isinf(self.best_val) else self.best_val,
            },
        )
        for fname, src in self.extra_files.items():
            if Path(src).is_file():
                (Path(out) / fname).write_bytes(Path(src).read_bytes())
        logger.info("Exported safetensors folder %s", out)
        return Path(out)

    # --------------------------------------------------------------- train
    def _amp(self):
        if self.use_bf16 and self.device.type in ("cuda", "cpu"):
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _next_batch(self, it):
        try:
            return next(it), it
        except StopIteration:
            it = iter(self.train_loader)
            return next(it), it

    def _prep(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch.get("targets", input_ids[:, 1:]).to(self.device, non_blocking=True)
        if targets.shape[1] != input_ids.shape[1]:
            input_ids = input_ids[:, :-1]
        return input_ids, targets

    def train(self) -> None:
        tc = self.cfg.training
        self.model.train()
        data_iter = iter(self.train_loader)
        pbar = tqdm(total=tc.max_steps, initial=self.global_step, desc="train")
        t0 = time.time()
        tokens_seen = 0

        while self.global_step < tc.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_acc, last_out, last_ids = 0.0, None, None

            for _ in range(self.accum):
                batch, data_iter = self._next_batch(data_iter)
                input_ids, targets = self._prep(batch)
                with self._amp():
                    out = self.model(
                        input_ids,
                        targets=targets,
                        lambda_err=tc.lambda_err,
                        lambda_bal=tc.lambda_bal,
                    )
                (out.loss / self.accum).backward()
                loss_acc += float(out.loss.detach()) / self.accum
                tokens_seen += input_ids.numel()
                last_out, last_ids = out, input_ids

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            if not torch.isfinite(grad_norm):
                # skip a poisoned step instead of corrupting the weights
                logger.warning("non-finite grad norm at step %d; skipping update", self.global_step)
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            pbar.update(1)
            m = last_out.metrics
            pbar.set_postfix(
                loss=f"{loss_acc:.4f}", ce=f"{m.get('ce', 0):.3f}", gate=f"{m.get('gate_mean', 0):.3f}"
            )

            # Slow-memory replay (paper: surprise-weighted, per-sequence)
            if self.model.meta.should_replay(self.global_step, tc.replay_every):
                if last_out.state is not None and "M" in last_out.state:
                    S = m.get("_seq_surprise")
                    if S is None or not torch.is_tensor(S):
                        S = torch.full(
                            (last_ids.size(0),),
                            float(m.get("mean_S", m.get("ce", 1.0))),
                            device=self.device,
                        )
                    self.model.replay_slow(last_out.state["M"], S, tc.replay_eta)

            if self.global_step % tc.log_every == 0:
                dt = max(1e-9, time.time() - t0)
                logger.info(
                    "step %d loss=%.4f lr=%.2e grad_norm=%.3f tok/s=%.0f metrics=%s",
                    self.global_step,
                    loss_acc,
                    self.scheduler.get_last_lr()[0],
                    float(grad_norm),
                    tokens_seen / dt,
                    {k: v for k, v in m.items() if not k.startswith("_")},
                )

            if self.val_loader is not None and self.global_step % tc.eval_every == 0:
                val_loss = self.evaluate()
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save_checkpoint("best")
                    self.export_safetensors("best")
                self.model.train()

            if self.global_step % tc.checkpoint_every == 0:
                self._save_checkpoint("latest")

        pbar.close()
        self._save_checkpoint("final")
        self.export_safetensors("final")
        logger.info("Training finished at step %d", self.global_step)

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            input_ids, targets = self._prep(batch)
            with self._amp():
                out = self.model(input_ids, targets=targets)
            total += float(out.metrics["ce"])  # pure cross-entropy => ppl is meaningful
            n += 1
        avg = total / max(1, n)
        logger.info("Validation CE: %.4f (ppl %.2f) on %d batches", avg, math.exp(min(avg, 20)), n)
        return avg
