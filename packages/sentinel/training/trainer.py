"""Training loop for SENTINEL (1B-ready).

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
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from sentinel.checkpoint.io import _torch_load, save_checkpoint
from sentinel.config import SENTINELConfig, save_config
from sentinel.model import SENTINEL
from sentinel.training.distributed import is_distributed, is_main

logger = logging.getLogger(__name__)

# Keep train logs short — full metric dumps fill Colab/disk fast
_LOG_KEYS = (
    "ce", "err_energy", "err_mid", "err_coarse", "gate_mean", "mean_S", "k_depth",
)


def _fmt_metrics(m: dict) -> str:
    """One compact line of key metrics (skip internal _* and bulky dumps)."""
    if not m:
        return "-"
    parts = []
    for k in _LOG_KEYS:
        if k not in m:
            continue
        v = m[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}" if abs(v) < 100 else f"{k}={v:.2e}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts) if parts else "-"



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


def _param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    lr: float,
    memory_lr_scale: float = 0.5,
) -> list[dict[str, Any]]:
    """Param groups: no decay on 1-D / embed; lower LR on memory + neuromodulator."""
    mem_prefixes = ("fast_mem.", "slow_mem.", "neuromod.", "P_f.", "P_s.", "mem_scale", "slow_scale")
    decay, no_decay, mem_decay, mem_no_decay, seen = [], [], [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_mem = name.startswith(mem_prefixes) or name in ("mem_scale", "slow_scale")
        is_no_decay = p.ndim < 2 or name.startswith(("embed.", "slow_mem.M_s"))
        if is_mem and is_no_decay:
            mem_no_decay.append(p)
        elif is_mem:
            mem_decay.append(p)
        elif is_no_decay:
            no_decay.append(p)
        else:
            decay.append(p)
    mem_lr = lr * float(memory_lr_scale)
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]
    if mem_decay:
        groups.append({"params": mem_decay, "weight_decay": weight_decay, "lr": mem_lr})
    if mem_no_decay:
        groups.append({"params": mem_no_decay, "weight_decay": 0.0, "lr": mem_lr})
    return groups


class Trainer:
    def __init__(
        self,
        model: SENTINEL,
        cfg: SENTINELConfig,
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
        self.dist = is_distributed()
        self.rank = torch.distributed.get_rank() if self.dist else 0
        self.world = torch.distributed.get_world_size() if self.dist else 1
        self.is_main = self.rank == 0
        if self.dist and torch.cuda.is_available():
            # one GPU per rank (torchrun sets LOCAL_RANK)
            self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        else:
            from sentinel.training.gpu import configure_accelerator

            _acc = configure_accelerator(
                tc.device if tc.device != "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"),
                bf16=True if getattr(tc, "precision", "fp32") == "bf16" else False,
                compile_model=bool(getattr(tc, "compile", False)),
                tf32=True,  # also sets TF32 + matmul precision on CUDA
            )
            self.device = _acc.device
        self.model.to(self.device)
        # Tensor / pipeline parallel groups (no-op when sizes == 1)
        from sentinel.training.parallelism import apply_tensor_parallel, init_parallel_groups
        self.parallel = init_parallel_groups(
            tensor_parallel_size=int(getattr(tc, "tensor_parallel_size", 1) or 1),
            pipeline_parallel_size=int(getattr(tc, "pipeline_parallel_size", 1) or 1),
        )
        if self.parallel.tp_size > 1:
            apply_tensor_parallel(self.model, self.parallel)
            self.model.to(self.device)
        # honour the config instead of the model's hardcoded default
        self.model._truncate_window = max(1, int(tc.truncate_write_window))

        if tc.precision not in ("fp32", "bf16"):
            raise ValueError(f"training.precision must be fp32|bf16, got {tc.precision!r}")
        self.use_bf16 = tc.precision == "bf16"
        self.accum = max(1, int(tc.grad_accum_steps))

        adamw_kw: dict[str, Any] = {"lr": tc.lr, "betas": (0.9, 0.95)}
        if self.device.type == "cuda":
            adamw_kw["fused"] = True  # one fused kernel for the whole update
        mem_scale = float(getattr(tc, "memory_lr_scale", 0.5) or 0.5)
        self.optimizer = AdamW(
            _param_groups(self.model, tc.weight_decay, tc.lr, mem_scale),
            **adamw_kw,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=tc.warmup_steps,
            num_training_steps=tc.max_steps,
        )
        # EMA of weights for stabler eval (0 = off)
        self.ema_decay = float(getattr(tc, "ema_decay", 0.0) or 0.0)
        self._ema: dict[str, Tensor] | None = None
        if self.ema_decay > 0.0:
            self._ema = {
                n: p.detach().float().clone()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        # Data parallel: wrap AFTER the optimizer is built from the plain module's parameters.
        # `self.model` stays the plain module (meta, replay, export); `self._fwd` is what we call.
        self._fwd: torch.nn.Module = self.model
        if self.dist:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self._fwd = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=False,  # True only if ablations leave params unused
            )
            torch.manual_seed(tc.seed + 1000 * self.rank)  # params were broadcast; dropout/data differ per rank

        self.output_dir = Path(tc.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_config(cfg, self.output_dir / "config.yaml")
        self._epoch = 0

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
            self._epoch += 1
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)  # DistributedSampler: new shuffle, still disjoint per rank
            it = iter(self.train_loader)
            return next(it), it

    def _embed_norm(self) -> float:
        return float(self.model.embed.weight.detach().float().norm(dim=-1).mean())

    def _prep(self, batch):
        input_ids = batch["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids)
        input_ids = input_ids.to(self.device, non_blocking=True)
        if "targets" in batch and batch["targets"] is not None:
            targets = batch["targets"]
            if not isinstance(targets, torch.Tensor):
                targets = torch.as_tensor(targets)
            targets = targets.to(self.device, non_blocking=True)
        else:
            targets = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
            return input_ids, targets
        if targets.shape[1] != input_ids.shape[1]:
            input_ids = input_ids[:, :-1]
        return input_ids, targets

    def train(self) -> None:
        tc = self.cfg.training
        self.model.train()
        data_iter = iter(self.train_loader)
        pbar = tqdm(total=tc.max_steps, initial=self.global_step, desc="train", disable=not self.is_main)
        t0 = time.time()
        tokens_seen = 0
        embed0 = self._embed_norm()  # Appendix B item 14: watch for collapse caused by the error loss

        # cache hierarchical base λ (model config) once
        base_mid = float(getattr(self.cfg.model, "lambda_err_mid", 0.05) or 0.0)
        base_coarse = float(getattr(self.cfg.model, "lambda_err_coarse", 0.02) or 0.0)
        warm_steps = int(getattr(tc, "lambda_err_warmup_steps", 0) or 0)
        log_every = max(1, int(tc.log_every))

        while self.global_step < tc.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            last_out, last_ids = None, None
            # accumulate loss on device — one .item() only when logging
            loss_acc_t = torch.zeros((), device=self.device)

            warm_scale = (
                min(1.0, (self.global_step + 1) / warm_steps) if warm_steps > 0 else 1.0
            )
            lam_err = tc.lambda_err * warm_scale
            lam_mid = base_mid * warm_scale
            lam_coarse = base_coarse * warm_scale
            do_log = (self.global_step + 1) % log_every == 0

            for micro in range(self.accum):
                batch, data_iter = self._next_batch(data_iter)
                input_ids, targets = self._prep(batch)
                sync_ctx = (
                    self._fwd.no_sync()
                    if (self.dist and micro < self.accum - 1)
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    with self._amp():
                        out = self._fwd(
                            input_ids,
                            targets=targets,
                            lambda_err=lam_err,
                            lambda_err_mid=lam_mid,
                            lambda_err_coarse=lam_coarse,
                            lambda_bal=tc.lambda_bal,
                            collect_stats=do_log,
                        )
                    (out.loss / self.accum).backward()
                loss_acc_t = loss_acc_t + out.loss.detach() / self.accum
                tokens_seen += input_ids.numel()
                last_out, last_ids = out, input_ids

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            if not torch.isfinite(grad_norm):
                logger.warning("non-finite grad norm at step %d; skipping update", self.global_step)
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                self._ema_update()
                self.scheduler.step()

            self.global_step += 1
            pbar.update(1)
            m = last_out.metrics if last_out is not None else {}
            if do_log:
                loss_acc = float(loss_acc_t.item())
                pbar.set_postfix(
                    loss=f"{loss_acc:.4f}",
                    ce=f"{m.get('ce', 0):.3f}",
                    gate=f"{m.get('gate_mean', 0):.3f}",
                )
            else:
                loss_acc = 0.0  # filled below only for log branch

            # Slow-memory replay (paper: surprise-weighted, per-sequence).
            # Data-parallel: the decision must be identical on every rank (replay all-reduces), so use the
            # fixed step schedule only; the surprise-budget trigger is rank-local.
            if self.dist:
                do_replay = tc.replay_every > 0 and self.global_step > 0 and self.global_step % tc.replay_every == 0
            else:
                do_replay = self.model.meta.should_replay(self.global_step, tc.replay_every)
            if do_replay:
                if last_out.state is not None and "M" in last_out.state:
                    S = m.get("_seq_surprise")
                    if S is None or not torch.is_tensor(S):
                        S = torch.full(
                            (last_ids.size(0),),
                            float(m.get("mean_S", m.get("ce", 1.0))),
                            device=self.device,
                        )
                    self.model.replay_slow(last_out.state["M"], S, tc.replay_eta)
                    # Reset surprise budget so consolidation does not fire every step
                    # after the first threshold crossing.
                    self.model.meta.acknowledge_replay()

            if self.is_main and do_log:
                loss_acc = float(loss_acc_t.item())
                dt = max(1e-9, time.time() - t0)
                # embed_norm only every 5th log line (extra host sync)
                en = self._embed_norm() if (self.global_step // log_every) % 5 == 0 else None
                if en is not None:
                    logger.info(
                        "step %d loss=%.4f lr=%.2e gn=%.2f tok/s=%.0f emb=%.3f %s",
                        self.global_step,
                        loss_acc,
                        self.scheduler.get_last_lr()[0],
                        float(grad_norm),
                        tokens_seen * self.world / dt,
                        en,
                        _fmt_metrics(m),
                    )
                else:
                    logger.info(
                        "step %d loss=%.4f lr=%.2e gn=%.2f tok/s=%.0f %s",
                        self.global_step,
                        loss_acc,
                        self.scheduler.get_last_lr()[0],
                        float(grad_norm),
                        tokens_seen * self.world / dt,
                        _fmt_metrics(m),
                    )
                # Append one compact JSONL row (easy to plot; tiny vs full dict logs)
                try:
                    import json
                    row = {
                        "step": self.global_step,
                        "loss": round(loss_acc, 6),
                        "lr": self.scheduler.get_last_lr()[0],
                        "tok_s": round(tokens_seen * self.world / dt, 1),
                    }
                    for k in _LOG_KEYS:
                        if k in m and isinstance(m[k], (int, float)):
                            row[k] = round(float(m[k]), 6)
                    log_path = self.output_dir / "train_metrics.jsonl"
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                except OSError:
                    pass
                if en is not None and embed0 > 0 and en < 0.3 * embed0:
                    logger.warning(
                        "embedding norm fell to %.0f%% of start (error-loss collapse risk)",
                        100 * en / embed0,
                    )

            # checkpoint / eval run on rank 0 only (plain module: no collectives, other ranks wait in all-reduce)
            if self.is_main and self.val_loader is not None and self.global_step % tc.eval_every == 0:
                val_loss = self.evaluate()
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save_checkpoint("best")
                    self.export_safetensors("best")
                self.model.train()

            if self.is_main and self.global_step % tc.checkpoint_every == 0:
                self._save_checkpoint("latest")

        pbar.close()
        if self.is_main:
            self._save_checkpoint("final")
            self.export_safetensors("final")
            logger.info("Training finished at step %d", self.global_step)
        if self.dist:
            torch.distributed.barrier()

    def _ema_update(self) -> None:
        if not self._ema:
            return
        d = self.ema_decay
        # Cached (name, param, ema_tensor) list — avoids named_parameters() scan every step
        pairs = getattr(self, "_ema_pairs", None)
        if pairs is None:
            pairs = [
                (n, p, self._ema[n])
                for n, p in self.model.named_parameters()
                if n in self._ema and p.requires_grad
            ]
            self._ema_pairs = pairs
        with torch.no_grad():
            for _n, p, buf in pairs:
                buf.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @contextlib.contextmanager
    def _ema_weights(self):
        """Temporarily swap model params to EMA for eval; restore after."""
        if not self._ema:
            yield
            return
        backup = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self._ema:
                    backup[n] = p.data.clone()
                    p.data.copy_(self._ema[n].to(device=p.device, dtype=p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in backup:
                        p.data.copy_(backup[n])

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        """Held-out CE in float32 (amp/EMA do not distort the metric)."""
        self.model.eval()
        total, n = 0.0, 0
        # EMA often lags / hurts mid-train CE; only use when explicitly enabled
        use_ema = bool(getattr(self.cfg.training, "eval_ema", False))
        ctx = self._ema_weights() if use_ema else contextlib.nullcontext()
        vocab = int(getattr(getattr(self.model, "cfg", None), "vocab_size", 0) or 0)
        with ctx:
            for i, batch in enumerate(self.val_loader):
                if i >= max_batches:
                    break
                input_ids, targets = self._prep(batch)
                # Forward in bf16/fp32 as trained, but CE always from float32 logits
                with self._amp():
                    out = self.model(
                        input_ids,
                        targets=targets,
                        collect_stats=False,
                        write_slow=True,  # same forward as training (slow ring ON)
                    )
                logits = out.logits.float()
                ce_t = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                total += float(ce_t.item())
                n += 1
        avg = total / max(1, n)
        ppl = math.exp(min(avg, 20.0))
        logger.info("Validation CE: %.4f (ppl %.2f) on %d batches", avg, ppl, n)
        if vocab > 0 and avg > math.log(vocab) + 1.0:
            logger.warning(
                "Val CE %.2f is worse than random (log vocab≈%.2f). "
                "Check val.bin / tokenizer match, or train longer with more data.",
                avg,
                math.log(vocab),
            )
        return avg
