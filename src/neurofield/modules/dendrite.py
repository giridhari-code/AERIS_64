"""Local causal dendrite with delayed prediction error injection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Dendrite(nn.Module):
    """
    Causal local window over recent embeddings + delayed error injection.

    d_t = tanh( sum_{j=0}^{w-1} c_j ⊙ x_{t-j} + b ) + γ ⊙ e_t
    """

    def __init__(self, d_model: int, window: int = 3):
        super().__init__()
        self.d_model = d_model
        self.window = window
        self.coeffs = nn.Parameter(torch.ones(window, d_model) / window)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.gamma = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(
        self,
        x_hist: Tensor,          # (B, W, D) most recent W embeddings, oldest first
        error: Tensor | None,    # (B, D) delayed prediction error e_t
    ) -> Tensor:
        """
        Args:
            x_hist: history of embeddings, shape (batch, window, d_model)
            error: previous-step prediction error, shape (batch, d_model) or None
        Returns:
            dendrite feature d_t of shape (batch, d_model)
        """
        # Weighted sum over causal window
        # coeffs: (W, D), x_hist: (B, W, D)
        weighted = (self.coeffs.unsqueeze(0) * x_hist).sum(dim=1)  # (B, D)
        base = torch.tanh(weighted + self.bias)
        if error is not None:
            base = base + self.gamma * error
        return base
