"""Surprise-gated write controller (neuromodulator)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class Neuromodulator(nn.Module):
    """
    Compute write gate g_t from relative surprise, state activity, and
    metacognitive encoding bias δ (paper Appendix A + brain-like tagging).

    g_t = σ( a_s (S̃_t - 1) + a_c |h|_mean + b + δ_t )
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.a_s = nn.Parameter(torch.tensor(3.0))
        self.a_c = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        relative_surprise: Tensor,   # (B,) S_t / bar_S
        h: Tensor,                   # (B, D)
        gate_cap: float = 1.0,
        gate_bias: Optional[Tensor] = None,  # (B,) from Metacognition; overrides internal δ
    ) -> Tensor:
        """Returns gate g of shape (B, 1)."""
        if gate_bias is None:
            # Paper discrete δ when meta does not supply bias
            delta = torch.zeros_like(relative_surprise)
            delta = torch.where(relative_surprise > 1.5, torch.full_like(delta, 0.5), delta)
            delta = torch.where(relative_surprise < 0.5, torch.full_like(delta, -0.5), delta)
        else:
            delta = gate_bias
            if delta.ndim > 1:
                delta = delta.reshape(delta.shape[0], -1).mean(-1)
            if delta.shape[0] != relative_surprise.shape[0]:
                delta = delta.mean().expand_as(relative_surprise)

        activity = h.abs().mean(dim=-1)  # (B,)
        logit = (
            self.a_s * (relative_surprise - 1.0)
            + self.a_c * activity
            + self.bias
            + delta
        )
        g = torch.sigmoid(logit).unsqueeze(-1)  # (B, 1)
        return g.clamp(max=gate_cap)
