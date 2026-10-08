"""Fast (delta-rule) and slow (consolidated) memory matrices — paper-faithful."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FastMemory(nn.Module):
    """
    Per-sequence fast weight matrix (Appendix A):

        k_t = normalize(W_k d_{t-1})
        v_t = W_v o_t
        M_t = Λ M_{t-1} + g_t * k_t (v_t - M_{t-1}^T k_t)^T

    Read:
        q_t = normalize(W_q d_t)
        r_t = M_t^T q_t

    M shape: (B, d_k, d_v)
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

        # Decay must survive a full recall sequence (≈ 2*n_pairs + 2 tokens).
        # Old init linspace(1, 6) → λ≈0.73..0.997: fast channels wiped in <10 steps.
        # New: λ ≈ 0.97 .. 0.998 so associations persist across the episode.
        self.log_lambda = nn.Parameter(torch.linspace(3.5, 6.5, d_k))

        nn.init.xavier_uniform_(self.W_k.weight)
        # Key/query alignment at init — retrieval works before W_q is trained.
        self.W_q.weight.data.copy_(self.W_k.weight.data)
        nn.init.xavier_uniform_(self.W_v.weight)

    @property
    def lambda_diag(self) -> Tensor:
        return torch.sigmoid(self.log_lambda)

    def write(
        self,
        M: Tensor,       # (B, d_k, d_v)
        d_prev: Tensor,  # (B, D) key from previous dendrite
        o_t: Tensor,     # (B, D) skill output → value
        g: Tensor,       # (B, 1) gate
    ) -> Tensor:
        # Skip meaningless write when previous dendrite is still the zero state
        # (first token of a sequence / fresh session).
        prev_norm = d_prev.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        active = (prev_norm > 1e-4).float()  # (B, 1)
        g_eff = g * active

        k = self.W_k(d_prev)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        v = self.W_v(o_t)
        pred = torch.einsum("bkd,bk->bd", M, k)
        delta = v - pred
        lam = self.lambda_diag.view(1, -1, 1)
        update = g_eff.unsqueeze(-1) * torch.einsum("bk,bd->bkd", k, delta)
        return lam * M + update

    def read(self, M: Tensor, d_t: Tensor) -> Tensor:
        q = self.W_q(d_t)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.einsum("bkd,bk->bd", M, q)


class SlowMemory(nn.Module):
    """
    Consolidated slow memory.

    Read: u = M_s^T q(d_t)  with q = W_q d_t
    Replay: move M_s toward surprise-weighted average of fast M matrices.
    M_s shape: (d_k, d_v) — shared across batch (slow weights).
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        # Zero init: random xavier M_s injects noise into logits before any replay
        # and masks the contribution of fast memory during early training.
        self.M_s = nn.Parameter(torch.zeros(d_k, d_v))
        nn.init.xavier_uniform_(self.W_q.weight)

    def read(self, d_t: Tensor) -> Tensor:
        q = self.W_q(d_t)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.einsum("kd,bk->bd", self.M_s, q)

    def replay(self, M: Tensor, surprises: Tensor, eta: float) -> None:
        """
        M: (B, d_k, d_v) last fast memory per sequence
        surprises: (B,) per-sequence surprise weights
        """
        if M.dim() != 3:
            return
        B = M.size(0)
        s = surprises.detach().reshape(-1).clamp(min=0.0)
        if s.numel() != B:
            s = s.mean().expand(B)
        w = s / (s.sum() + 1e-8)
        avg = (w.view(B, 1, 1) * M.detach()).sum(0)  # (d_k, d_v)
        self.M_s.data.mul_(1.0 - eta).add_(eta * avg)
