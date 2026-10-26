"""Top-k skill router with straight-through estimator + optional workspace bias."""

from __future__ import annotations

from typing import Optional

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
    π = softmax(W_r h + workspace_bias)
    w = softmax(top-k logits)
    w_STE = w + π - sg(π)
    o = Σ w_STE_i * Op_i(h)

    Optional ``workspace`` (J-space broadcast) biases skill selection toward
    whatever is currently "on mind" — more intelligent routing under access
    consciousness.
    """

    def __init__(self, d_model: int, n_skills: int, top_k: int, jspace_dim: int = 0):
        super().__init__()
        self.n_skills = n_skills
        self.top_k = min(top_k, n_skills)
        self.router = nn.Linear(d_model, n_skills, bias=False)
        self.skills = nn.ModuleList([SkillOperator(d_model) for _ in range(n_skills)])

        self.jspace_dim = int(jspace_dim)
        if self.jspace_dim > 0:
            self.ws_bias = nn.Linear(self.jspace_dim, n_skills, bias=False)
            nn.init.zeros_(self.ws_bias.weight)  # start neutral
        else:
            self.ws_bias = None

    def forward(
        self,
        h: Tensor,
        workspace: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.router(h)  # (B, n_skills)

        if workspace is not None and self.ws_bias is not None:
            ws = workspace
            if ws.ndim == 1:
                ws = ws.unsqueeze(0).expand(h.size(0), -1)
            if ws.size(-1) != self.jspace_dim:
                if ws.size(-1) > self.jspace_dim:
                    ws = ws[..., : self.jspace_dim]
                else:
                    pad = ws.new_zeros(*ws.shape[:-1], self.jspace_dim)
                    pad[..., : ws.size(-1)] = ws
                    ws = pad
            logits = logits + 0.5 * self.ws_bias(ws)

        pi = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        w = torch.zeros_like(logits)
        src = F.softmax(topk_vals, dim=-1).to(dtype=w.dtype)
        w.scatter_(-1, topk_idx, src)

        w_ste = w + pi - pi.detach()

        if self.n_skills > 16 and self.top_k < self.n_skills:
            outs = h.new_zeros(h.size(0), self.n_skills, h.size(-1))
            for s in topk_idx.reshape(-1).unique().tolist():
                outs[:, s] = self.skills[s](h)
        else:
            outs = torch.stack([op(h) for op in self.skills], dim=1)
        o = torch.einsum("bs,bsd->bd", w_ste, outs)
        return o, pi, w
