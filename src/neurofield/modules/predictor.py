"""Next-input predictor used for surprise and error energy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Predictor(nn.Module):
    """
    Predicts the next embedding from the current field state.

    x̂_{t+1} = f_p(h_t)
    e_{t+1}  = x_{t+1} - x̂_{t+1}
    S_{t+1}  = sg(mean |e_{t+1}|)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)

    @staticmethod
    def compute_error_and_surprise(
        pred: Tensor,          # (B, D) prediction of x_{t+1}
        target: Tensor,        # (B, D) actual x_{t+1}
        detach_target: bool = True,
    ) -> tuple[Tensor, Tensor]:
        # Detach target so error energy trains the predictor toward embeddings
        # without pulling the embedding table into a collapse with the tied head.
        tgt = target.detach() if detach_target else target
        error = tgt - pred
        # Surprise is detached (no gradient through S)
        surprise = torch.mean(error.abs(), dim=-1).detach()  # (B,)
        return error, surprise
