"""Next-input predictor used for surprise and error energy — stronger residual form."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Predictor(nn.Module):
    """
    Predicts the next embedding from the current field state.

    x̂_{t+1} = h + MLP(LN(h))     (residual — more stable, better gradients)
    e_{t+1}  = x_{t+1} - x̂_{t+1}
    S_{t+1}  = sg(mean |e_{t+1}|)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        # Start near identity residual so early training is stable
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: Tensor) -> Tensor:
        return h + self.net(self.norm(h))

    @staticmethod
    def compute_error_and_surprise(
        pred: Tensor,
        target: Tensor,
        detach_target: bool = True,
    ) -> tuple[Tensor, Tensor]:
        tgt = target.detach() if detach_target else target
        error = tgt - pred
        surprise = error.detach().abs().mean(dim=-1)
        return error, surprise
