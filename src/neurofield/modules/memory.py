"""Fast (delta-rule) and slow (context-window ring) memory — AESC / NeuroField.

Fast memory M  — short segment, written every token (delta rule).
Slow memory    — ring of slots covering the context window:
                 n_slots ≈ context_window / segment_len
                 segment end → snapshot fast M into next slot
                 read → content-address over all filled slots
"""

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
    Context-window handler (slow path).

    Fast memory = short segment (delta-rule within a few dozen tokens).
    Slow memory = ring of segment slots covering the **context window**:

        n_slots ≈ context_window / segment_len

    At each segment boundary the current fast matrix M is written into the
    next ring slot. Read content-addresses over all filled slots so long
    context is retrieved here instead of stretching fast M.

    State tensors (per batch sequence, carried in model state — not weights):
        slots: (B, n_slots, d_k, d_v)
        ptr:   int  (next write index)
        filled:(B, n_slots)  0/1

    Learned: W_q (query). Prefer tying to FastMemory.W_k at init.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, n_slots: int = 8):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_slots = max(1, int(n_slots))
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        # Optional global prior (still updated by classic replay for compat)
        self.M_s = nn.Parameter(torch.zeros(d_k, d_v))
        nn.init.xavier_uniform_(self.W_q.weight)

    def init_slots(self, batch: int, device: torch.device) -> tuple[Tensor, int, Tensor]:
        slots = torch.zeros(batch, self.n_slots, self.d_k, self.d_v, device=device)
        filled = torch.zeros(batch, self.n_slots, device=device)
        return slots, 0, filled

    def write_slot(
        self,
        slots: Tensor,       # (B, S, d_k, d_v)
        ptr: int,
        filled: Tensor,      # (B, S)
        M: Tensor,           # (B, d_k, d_v) fast memory snapshot
        weight: Tensor | None = None,  # (B,) optional strength
    ) -> tuple[Tensor, int, Tensor]:
        """Consolidate one segment of context into the ring (context window)."""
        B, S, _, _ = slots.shape
        if M.dim() != 3 or M.size(0) != B:
            return slots, ptr, filled
        w = 1.0
        if weight is not None:
            w = weight.detach().reshape(B, 1, 1).clamp(min=0.0)
            # normalize lightly
            w = w / (w.mean() + 1e-8)
        new_slots = slots.clone()
        new_filled = filled.clone()
        snap = M.detach() * (w if isinstance(w, Tensor) else 1.0)
        new_slots[:, ptr] = snap
        new_filled[:, ptr] = 1.0
        new_ptr = (ptr + 1) % S
        return new_slots, new_ptr, new_filled

    def read(
        self,
        d_t: Tensor,
        slots: Tensor | None = None,
        filled: Tensor | None = None,
    ) -> Tensor:
        """
        Query context window slots + small global prior M_s.
        If slots is None, falls back to classic M_s-only read.
        """
        q = self.W_q(d_t)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        # Global prior
        u_prior = torch.einsum("kd,bk->bd", self.M_s, q)

        if slots is None or filled is None:
            return u_prior

        # Per-slot read: r_s = M_s^T q  -> (B, S, d_v)
        r = torch.einsum("bskd,bk->bsd", slots, q)
        # Slot keys = mean over value dim -> (B, S, d_k)
        k_slot = slots.mean(dim=-1)
        scores = torch.einsum("bk,bsk->bs", q, k_slot)
        # Mask empty slots
        scores = scores.masked_fill(filled <= 0, -1e9)
        # If nothing filled, only prior
        any_fill = filled.sum(dim=-1, keepdim=True)  # (B,1)
        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.where(any_fill > 0, alpha, torch.zeros_like(alpha))
        u_ctx = torch.einsum("bs,bsd->bd", alpha, r)
        # Context window dominates when slots are filled; else global prior only
        gate = (any_fill > 0).float()  # (B,1)
        return gate * (0.85 * u_ctx + 0.15 * u_prior) + (1.0 - gate) * u_prior

    def replay(self, M: Tensor, surprises: Tensor, eta: float) -> None:
        """Legacy global prior update (batch-level consolidation)."""
        if M.dim() != 3:
            return
        B = M.size(0)
        s = surprises.detach().reshape(-1).clamp(min=0.0)
        if s.numel() != B:
            s = s.mean().expand(B)
        w = s / (s.sum() + 1e-8)
        avg = (w.view(B, 1, 1) * M.detach()).sum(0)
        self.M_s.data.mul_(1.0 - eta).add_(eta * avg)

    def tie_query_to(self, W_k: nn.Linear) -> None:
        """Align slow query basis with fast key basis (context retrieval match)."""
        with torch.no_grad():
            self.W_q.weight.copy_(W_k.weight)
