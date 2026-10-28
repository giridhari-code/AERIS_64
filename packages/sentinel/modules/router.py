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

    This is **not** a different model family (still AERIS / SENTINEL).
    ``sparse_skill_exec`` is only an implementation switch for which skill
    MLPs are executed; the top-k router equations are the original design.
    """

    def __init__(
        self,
        d_model: int,
        n_skills: int,
        top_k: int,
        jspace_dim: int = 0,
        sparse_skill_exec: bool = True,
    ):
        super().__init__()
        self.n_skills = n_skills
        self.top_k = min(top_k, n_skills)
        # Execution policy only — routing math (π, top-k w, STE) is unchanged AERIS design
        self.sparse_skill_exec = bool(sparse_skill_exec)
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
            elif ws.ndim == 2 and ws.size(0) != h.size(0):
                b_h, b_w = h.size(0), ws.size(0)
                if b_w > 0 and b_h % b_w == 0:
                    t_steps = b_h // b_w
                    ws = ws.unsqueeze(1).expand(b_w, t_steps, -1).reshape(b_h, -1)
                elif b_h < b_w:
                    ws = ws[:b_h]
                else:
                    reps = (b_h + b_w - 1) // b_w
                    ws = ws.repeat(reps, 1)[:b_h]
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

        # AERIS router is already top-k gated (w = softmax over top-k logits).
        # sparse_skill_exec only skips *running* non-selected skill MLPs (compute).
        # Math stays: o = Σ w_STE_i * Op_i(h) with full π for the router.
        use_sparse = self.sparse_skill_exec and self.top_k < self.n_skills
        if not use_sparse:
            outs = torch.stack([op(h) for op in self.skills], dim=1)
            o = torch.einsum("bs,bsd->bd", w_ste, outs)
        else:
            # One host sync (tolist) instead of unique() + tolist(): unique() has a data-dependent
            # output size (its own sync) and launches a radix sort. Same ascending order -> same result.
            used = sorted(set(topk_idx.reshape(-1).tolist()))
            if len(used) <= max(4, self.top_k * 2):
                o = h.new_zeros(h.size(0), h.size(-1))
                for s in used:
                    o = o + w_ste[:, s].unsqueeze(-1) * self.skills[s](h)
            else:
                outs = h.new_zeros(h.size(0), self.n_skills, h.size(-1))
                for s in used:
                    outs[:, s] = self.skills[s](h)
                o = torch.einsum("bs,bsd->bd", w_ste, outs)
        return o, pi, w
