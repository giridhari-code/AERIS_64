"""k-step recurrent neural field with memory read and local recurrence.

Paper Appendix A:
  h ← (1-ζ)⊙h + ζ⊙LN( α⊙h + β⊙(d_t + R(tanh h) + W_m r_{t-1}) )

R is a shared local convolution over neighbouring units (neuron index).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NeuralField(nn.Module):
    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.beta = nn.Parameter(torch.ones(d_model) * 0.1)
        self.zeta = nn.Parameter(torch.ones(d_model) * 0.5)
        self.W_m = nn.Linear(d_mem, d_model, bias=False)
        # Shared local conv over neuron index (length = d_model)
        self.R = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.R.weight)
        self.R.weight.data[0, 0, 1] = 1.0  # identity centre; sides learn
        self.norm = nn.LayerNorm(d_model)

    def single_step(
        self,
        h: Tensor,  # (B, D)
        d_t: Tensor,
        r_prev: Tensor,  # (B, d_mem)
    ) -> Tensor:
        # R(tanh h): conv along neuron dimension
        act = torch.tanh(h).unsqueeze(1)  # (B, 1, D)
        local = self.R(act).squeeze(1)  # (B, D)
        mem = self.W_m(r_prev)
        pre = self.alpha * h + self.beta * (d_t + local + mem)
        zeta = self.zeta.clamp(0.0, 1.0)
        return (1.0 - zeta) * h + zeta * self.norm(pre)

    def forward(self, h: Tensor, d_t: Tensor, r_prev: Tensor, k: int) -> Tensor:
        for _ in range(max(1, int(k))):
            h = self.single_step(h, d_t, r_prev)
        return h
