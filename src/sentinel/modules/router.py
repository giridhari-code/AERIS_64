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
        logits = self.router(h)  # (B, n_skills)
        pi = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        # zeros_like under autocast can be fp32 while softmax(topk) is bf16 —
        # scatter requires matching dtypes.
        w = torch.zeros_like(logits)
        src = F.softmax(topk_vals, dim=-1).to(dtype=w.dtype)
        w.scatter_(-1, topk_idx, src)

        w_ste = w + pi - pi.detach()

        outs = torch.stack([op(h) for op in self.skills], dim=1)
        o = torch.einsum("bs,bsd->bd", w_ste, outs)
        return o, pi, w
