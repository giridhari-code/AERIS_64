"""Local causal dendrite with delayed prediction error injection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Dendrite(nn.Module):
    """
    Causal local window over *past* embeddings + delayed error injection.

    d_t = tanh( sum_{j=0}^{W-1} c_j ⊙ x_{t-1-j} + b ) + γ ⊙ e_t

    The window is strictly causal: it uses the previous W embeddings only
    (oldest first). The current token x_t is NOT included. This matches the
    sequential ``forward`` path and keeps train (T>1 parallel) identical to
    decode (T=1).
    """

    def __init__(self, d_model: int, window: int = 3):
        super().__init__()
        self.d_model = d_model
        self.window = window
        self.coeffs = nn.Parameter(torch.ones(window, d_model) / window)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.gamma = nn.Parameter(torch.ones(d_model) * 0.1)

    def window_features(self, xp: Tensor, T: int) -> Tensor:
        """Token-parallel causal window: tanh(sum_j c_j * past + b) for all t.

        ``xp`` = cat([x_hist (B, W, D), X (B, T, D)], dim=1) → length W+T.

        For position t the window is xp[t : t+W] (the W tokens *before* X_t):
          t=0 → full x_hist
          t=1 → x_hist[1:] + X[0]
          ...
        Never includes the current embedding X_t. Matches ``forward()``.
        """
        W = self.window
        need = W + T
        if xp.size(1) < need:
            raise ValueError(
                f"dendrite.window_features: xp length {xp.size(1)} < W+T={need} "
                f"(pass cat([x_hist, X], dim=1))"
            )
        # Causal past window (no +1 shift). Previously this started at index 1,
        # which skipped the oldest history token and incorrectly pulled in X_t.
        acc = self.coeffs[0] * xp[:, 0:T]
        for j in range(1, W):
            acc = acc + self.coeffs[j] * xp[:, j : j + T]
        return torch.tanh(acc + self.bias)

    def with_error(self, base: Tensor, error: Tensor | None) -> Tensor:
        return base if error is None else base + self.gamma * error

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
        # Weighted sum over causal window (past only — same as window_features)
        # coeffs: (W, D), x_hist: (B, W, D)
        weighted = (self.coeffs.unsqueeze(0) * x_hist).sum(dim=1)  # (B, D)
        base = torch.tanh(weighted + self.bias)
        if error is not None:
            base = base + self.gamma * error
        return base
