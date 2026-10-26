"""Multi-level next-input predictors — brain-like hierarchical prediction error.

Human cortex predicts at several timescales at once (fine features → short
phrases → coarser structure). When a guess is wrong, that level's error
drives learning.

SENTINEL mirrors this with three residual predictors on the same field state:

  fine   (Δ=1)  — next embedding x_{t+1}
  mid    (Δ=4)  — short-horizon target (mean of next 4 embeds)
  coarse (Δ=16) — longer-horizon target (mean of next 16 embeds)

Surprise for gating still uses the **fine** error (fast, every token).
Training adds mid/coarse MSE so weights learn multi-scale structure.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class Predictor(nn.Module):
    """Single-timescale residual predictor: x̂ = h + MLP(LN(h))."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
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


class HierarchicalPredictor(nn.Module):
    """Three-timescale predictors sharing field state h.

    forward(h) → fine next-step pred (API-compatible with Predictor).
    level_errors(H, X, ...) → fine/mid/coarse MSE for the train loss.
    """

    def __init__(
        self,
        d_model: int,
        *,
        mid_horizon: int = 4,
        coarse_horizon: int = 16,
        use_mid: bool = True,
        use_coarse: bool = True,
    ):
        super().__init__()
        self.mid_horizon = max(2, int(mid_horizon))
        self.coarse_horizon = max(self.mid_horizon + 1, int(coarse_horizon))
        self.use_mid = bool(use_mid)
        self.use_coarse = bool(use_coarse)

        self.fine = Predictor(d_model)
        self.mid = Predictor(d_model) if self.use_mid else None
        self.coarse = Predictor(d_model) if self.use_coarse else None

    def forward(self, h: Tensor) -> Tensor:
        return self.fine(h)

    @staticmethod
    def _future_mean(X: Tensor, horizon: int) -> Tensor:
        """For each t, mean(X[t+1 : t+1+horizon]); pad with last token."""
        B, T, D = X.shape
        if T == 0:
            return X
        pad = X[:, -1:, :].expand(B, horizon, D)
        xp = torch.cat([X, pad], dim=1)
        fut = xp[:, 1 : T + horizon, :]
        windows = fut.unfold(1, horizon, 1)[:, :T]  # (B, T, D, horizon)
        return windows.mean(dim=-1)

    def level_errors(
        self,
        H: Tensor,
        X: Tensor,
        *,
        X_hat_fine_shift: Optional[Tensor] = None,
        need_fine: bool = False,
        need_mid: bool = True,
        need_coarse: bool = True,
    ) -> dict[str, Tensor]:
        """Multi-level errors. Skip unused levels to save train compute."""
        B, T, D = X.shape
        X_det = X.detach()
        out: dict[str, Tensor] = {}
        zero = H.new_zeros(())

        if need_fine:
            if X_hat_fine_shift is not None:
                E_fine = X_det - X_hat_fine_shift
            else:
                pred = self.fine(H.reshape(B * T, -1)).view(B, T, -1)
                pred_shift = torch.cat([pred[:, :1] * 0, pred[:, :-1]], dim=1)
                E_fine = X_det - pred_shift
            out["E_fine"] = E_fine
            out["err_fine"] = E_fine.pow(2).mean()
        else:
            out["err_fine"] = zero

        flat = H.reshape(B * T, -1) if (need_mid or need_coarse) and T > 1 else None
        if need_mid and self.mid is not None and T > 1 and flat is not None:
            tgt = self._future_mean(X_det, self.mid_horizon)
            out["err_mid"] = (tgt - self.mid(flat).view(B, T, -1)).pow(2).mean()
        else:
            out["err_mid"] = zero

        if need_coarse and self.coarse is not None and T > 1 and flat is not None:
            tgt = self._future_mean(X_det, self.coarse_horizon)
            out["err_coarse"] = (tgt - self.coarse(flat).view(B, T, -1)).pow(2).mean()
        else:
            out["err_coarse"] = zero

        return out
