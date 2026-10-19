"""Minimal LoRA adapters for SENTINEL linear layers (preference / SFT efficient tuning)."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap nn.Linear with low-rank update: W' = W + (B @ A) * scale."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / max(1, rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x: (..., in) → delta
        x_d = self.dropout(x)
        delta = F.linear(F.linear(x_d, self.lora_A), self.lora_B) * self.scale
        return base_out + delta


def _should_adapt(name: str, module: nn.Module, target_modules: Iterable[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    targets = tuple(target_modules)
    if not targets or targets == ("all",):
        return True
    return any(t in name for t in targets)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("router", "head", "P_f", "P_s", "W_"),
) -> list[str]:
    """Replace matching Linear modules with LoRALinear. Returns adapted param names."""
    adapted: list[str] = []
    for name, module in list(model.named_modules()):
        if not _should_adapt(name, module, target_modules):
            continue
        parent_path = name.rsplit(".", 1)
        if len(parent_path) == 1:
            parent, attr = model, name
        else:
            parent_name, attr = parent_path
            parent = model.get_submodule(parent_name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        adapted.append(name)
    return adapted


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def merge_lora_(model: nn.Module) -> int:
    """Fold LoRA weights into base Linear in-place; returns number merged."""
    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        with torch.no_grad():
            # W <- W + B @ A * scale
            update = (module.lora_B @ module.lora_A) * module.scale
            module.base.weight.add_(update)
        parent_path = name.rsplit(".", 1)
        if len(parent_path) == 1:
            parent, attr = model, name
        else:
            parent = model.get_submodule(parent_path[0])
            attr = parent_path[1]
        setattr(parent, attr, module.base)
        n += 1
    return n
