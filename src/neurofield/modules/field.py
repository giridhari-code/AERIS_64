"""k-step recurrent neural field with memory read and local recurrence."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NeuralField(nn.Module):
    """
    Recurrent field updated k times per token:

    h ← (1-ζ)⊙h + ζ⊙LN( α⊙h + β⊙(d_t + R(tanh h) + W_m r_{t-1}) )
    """

    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.beta = nn.Parameter(torch.ones(d_model) * 0.1)
        self.zeta = nn.Parameter(torch.ones(d_model) * 0.5)
        self.W_m = nn.Linear(d_mem, d_model, bias=False)
        # Local recurrent convolution R over neighbouring units
        self.R = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1)
        self.norm = nn.LayerNorm(d_model)

    def single_step(
        self,
        h: Tensor,          # (B, D)
        d_t: Tensor,        # (B, D)
        r_prev: Tensor,     # (B, d_mem)
    ) -> Tensor:
        local = self.R(torch.tanh(h).unsqueeze(-1)).squeeze(-1)  # (B, D)
        mem = self.W_m(r_prev)
        pre = self.alpha * h + self.beta * (d_t + local + mem)
        updated = (1.0 - self.zeta) * h + self.zeta * self.norm(pre)
        return updated

    def forward(
        self,
        h: Tensor,
        d_t: Tensor,
        r_prev: Tensor,
        k: int,
    ) -> Tensor:
        """Run the field for k steps."""
        for _ in range(max(1, k)):
            h = self.single_step(h, d_t, r_prev)
        return h
