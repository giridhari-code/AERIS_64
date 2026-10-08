"""Top-k skill router with straight-through estimator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SkillOperator(nn.Module):
    """Simple residual MLP skill."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, h: Tensor) -> Tensor:
        return h + self.net(h)


class SkillRouter(nn.Module):
    """
    π = softmax(W_r h)
    w = softmax(top-k logits)
    w_STE = w + π - sg(π)          # forward = w, backward = ∇π
    o = Σ w_STE_i * Op_i(h)
    """

    def __init__(self, d_model: int, n_skills: int, top_k: int):
        super().__init__()
        self.n_skills = n_skills
        self.top_k = min(top_k, n_skills)
        self.router = nn.Linear(d_model, n_skills, bias=False)
        self.skills = nn.ModuleList([SkillOperator(d_model) for _ in range(n_skills)])

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            o_t: skill mixture (B, D)
            router_probs: full softmax π (B, n_skills)
            selected_weights: the top-k weights used in forward (B, n_skills)
        """
        logits = self.router(h)                      # (B, n_skills)
        pi = F.softmax(logits, dim=-1)

        # Hard top-k selection for forward weights
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        # Build sparse weight vector
        w = torch.zeros_like(logits)
        w.scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))

        # Straight-through estimator
        w_ste = w + pi - pi.detach()

        # Weighted skill outputs
        outs = torch.stack([op(h) for op in self.skills], dim=1)  # (B, n_skills, D)
        o = torch.einsum("bs,bsd->bd", w_ste, outs)
        return o, pi, w
