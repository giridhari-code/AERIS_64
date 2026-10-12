"""AERIS core model — default backbone is a decoder-only Transformer (GPT-style).

Legacy NeuroField modules remain under neurofield.modules for research/ablation;
production train + serve use causal Transformer by default (ModelConfig.architecture).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig
from neurofield.modules.transformer import TransformerStack


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Optional[Tensor] = None
    metrics: dict[str, float] = field(default_factory=dict)
    audit: Optional[dict[str, Any]] = None
    state: Optional[dict[str, Any]] = None


class NeuroField(nn.Module):
    """Public model class used by train + serve.

    Default: Transformer LM with causal attention mask.
    Set ModelConfig.architecture = "neurofield" only if you wire the legacy stack.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        safety: SafetyConfig | None = None,
        ablation: AblationConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.safety = safety or SafetyConfig()
        self.ablation = ablation or AblationConfig()
        arch = getattr(cfg, "architecture", "transformer") or "transformer"
        self.architecture = arch.lower().strip()

        n_layers = int(getattr(cfg, "n_layers", 4) or 4)
        n_heads = int(getattr(cfg, "n_heads", 4) or 4)
        # ensure d_model divisible by n_heads
        if cfg.d_model % n_heads != 0:
            # fall back to largest divisor
            for h in range(n_heads, 0, -1):
                if cfg.d_model % h == 0:
                    n_heads = h
                    break
        ffn_mult = int(getattr(cfg, "ffn_mult", 4) or 4)

        self.backbone = TransformerStack(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=cfg.max_seq_len,
            ffn_mult=ffn_mult,
            dropout=cfg.dropout,
            tie_embeddings=bool(getattr(cfg, "tie_embeddings", True)),
        )
        self._n_layers = n_layers
        self._n_heads = n_heads

    def set_ablation(self, ablation: AblationConfig) -> None:
        self.ablation = ablation

    def export_slow_state(self, state: Optional[dict]) -> Optional[dict[str, Tensor]]:
        return None

    def new_state(
        self, batch: int, device: torch.device, slow: Optional[dict[str, Tensor]] = None
    ) -> dict:
        return self._init_state(batch, device)

    def _init_state(self, batch: int, device: torch.device) -> dict[str, Any]:
        return {
            "architecture": "transformer",
            "memory_frozen": False,
        }

    def freeze_writes(self) -> None:
        """No-op for Transformer (no fast-memory gate)."""
        return

    def replay_slow(self, M: Tensor, surprises: Tensor, eta: float) -> None:
        return

    def reset_meta(self) -> None:
        return

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
        state: Optional[dict] = None,
        return_audit: bool = False,
        lambda_err: float = 0.0,
        lambda_bal: float = 0.0,
        write_slow: bool = True,
        k_depth: Optional[int] = None,
        collect_stats: bool = True,
        label_smoothing: float = 0.0,
    ) -> ModelOutput:
        """Causal Transformer forward. Extra kwargs kept for API compatibility."""
        if state is None:
            state = self._init_state(input_ids.size(0), input_ids.device)

        logits = self.backbone(input_ids)

        loss = None
        metrics: dict[str, float] = {
            "path": "transformer",
            "n_layers": float(self._n_layers),
            "n_heads": float(self._n_heads),
        }
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                label_smoothing=float(max(0.0, min(0.5, label_smoothing))),
            )
            loss = ce
            metrics["ce"] = float(ce.detach().item())

        audit = None
        if return_audit:
            audit = {"path": "transformer", "architecture": "transformer"}

        new_state = {**state, "architecture": "transformer"}
        return ModelOutput(
            logits=logits, loss=loss, metrics=metrics, audit=audit, state=new_state
        )

    def param_report(self) -> dict[str, int]:
        report: dict[str, int] = {}
        seen: set[int] = set()
        unique_total = 0
        for name, module in self.named_children():
            n = 0
            for p in module.parameters():
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    n += p.numel()
                    unique_total += p.numel()
            report[name] = n
        report["total"] = unique_total
        return report
