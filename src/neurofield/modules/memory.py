"""Fast (delta-rule) and slow (context-window ring) memory — AESC / NeuroField.

Fast memory M  — short segment, written every token (delta rule).
Slow memory    — ring of slots covering the context window:
                 n_slots ≈ context_window / segment_len
                 segment end → snapshot fast M into next slot
                 read → content-address over ALL filled slots
                 → long context is handled here, not by stretching fast M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FastMemory(nn.Module):
    """
    Per-sequence fast weight matrix (Appendix A):

        k_t = normalize(W_k d_{t-1})
        v_t = W_v o_t
        # Paper Appendix A.2 Change 5 — decay-consistent delta rule:
        #   M^Λ = Λ M_{t-1}
        #   M_t = M^Λ + g_t k_t (v_t - (M^Λ)^T k_t)^T

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

        # Decay must survive a full segment (≈ truncate window tokens).
        self.log_lambda = nn.Parameter(torch.linspace(3.5, 6.5, d_k))

        nn.init.xavier_uniform_(self.W_k.weight)
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
        prev_norm = d_prev.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        active = (prev_norm > 1e-4).float()
        g_eff = g * active

        k = self.W_k(d_prev)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        v = self.W_v(o_t)
        # Change 5: residual against *decayed* memory (not raw M)
        lam = self.lambda_diag.view(1, -1, 1)
        M_decay = lam * M
        pred = torch.einsum("bkd,bk->bd", M_decay, k)
        delta = v - pred
        update = g_eff.unsqueeze(-1) * torch.einsum("bk,bd->bkd", k, delta)
        return M_decay + update

    def read(self, M: Tensor, d_t: Tensor) -> Tensor:
        q = self.W_q(d_t)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.einsum("bkd,bk->bd", M, q)

    def write_seq(
        self,
        M: Tensor,              # (B, d_k, d_v) initial
        D: Tensor,              # (B, T, d_model) dendrite keys at each t
        O: Tensor,              # (B, T, d_model) skill values
        G: Tensor,              # (B, T, 1) gates
        d_prev0: Tensor | None = None,  # (B, d_model) key before first token (carry)
        truncate_window: int = 0,
        max_memory_norm: float = 0.0,
        segment_len: int = 0,
        seg_cnt0: int = 0,
        surprises: Tensor | None = None,  # (B, T) optional for ring snaps
    ) -> tuple[Tensor, Tensor, list, int]:
        """Delta-rule scan — looped-transformer time step inside a depth pass.

        Like a looped Transformer block applied over depth, each pass shares
        W_k/W_v/W_q. Projections are batched over T; the rank-1 memory update
        is a causal scan chunked by ``truncate_window`` (same role as a
        recurrent state in looped / recurrent Transformers).

        Returns (M_final, R, snaps, seg_cnt).
        """
        B, T, _ = D.shape
        if T == 0:
            return M, D.new_zeros(B, 0, self.d_v), [], int(seg_cnt0)

        if d_prev0 is None:
            d_prev0 = D.new_zeros(B, D.size(-1))
        d_prev = torch.cat([d_prev0.unsqueeze(1), D[:, :-1]], dim=1)

        k = self.W_k(d_prev)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        v = self.W_v(O)
        q = self.W_q(D)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        prev_norm = d_prev.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        g_eff = G * (prev_norm > 1e-4).float()
        lam = self.lambda_diag.view(1, -1, 1)

        # Looped-transformer style: depth shares weights; time is a scan over
        # precomputed (k,v,q). Chunk at truncate_window (BPTT boundary).
        reads: list[Tensor] = []
        snaps: list[tuple[Tensor, Tensor]] = []
        tw = max(1, int(truncate_window)) if truncate_window else T
        seg_len = max(0, int(segment_len))
        seg_cnt = int(seg_cnt0)
        max_n = float(max_memory_norm or 0.0)

        t = 0
        while t < T:
            t_end = min(T, t + tw)
            # one BPTT chunk — sequential rank-1 updates only (linear recurrence)
            for ti in range(t, t_end):
                kt = k[:, ti]
                M_decay = lam * M
                pred = torch.einsum("bkd,bk->bd", M_decay, kt)
                delta = v[:, ti] - pred
                M = M_decay + g_eff[:, ti].unsqueeze(-1) * torch.einsum("bk,bd->bkd", kt, delta)
                if max_n > 0:
                    mn = M.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                    M = M * (max_n / mn).clamp(max=1.0)
                reads.append(torch.einsum("bkd,bk->bd", M, q[:, ti]))
                if seg_len > 0:
                    seg_cnt += 1
                    if seg_cnt >= seg_len:
                        s_t = surprises[:, ti].detach() if surprises is not None else M.new_ones(B)
                        snaps.append((M.detach(), s_t))
                        seg_cnt = 0
            t = t_end
            if t < T:
                M = M.detach()  # truncate BPTT between chunks
        return M, torch.stack(reads, dim=1), snaps, seg_cnt


class SlowMemory(nn.Module):
    """
    Context-window handler (slow path) — ring of slots.

        n_slots ≈ context_window / segment_len

    Write (segment end or EOS):
        slots[ptr] ← snapshot(fast M)
        ptr ← (ptr + 1) % n_slots

    When the ring is full (``compact_overflow``) the oldest slot is NOT erased: it is folded into
    its neighbour as a count-weighted mean, so old context is diluted, never dropped. ``filled``
    then holds how many snapshots each slot represents (>= 1; masking still uses ``> 0``), and
    ``filled.sum()`` always equals the total number of writes.

    Read (every token):
        query all filled slots with q = normalize(W_q d_t)
        softmax over slot scores → weighted mix of slot reads
        (+ small global prior M_s from training replay)

    Long context is retrieved from the ring; fast M only covers the
    current short segment.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, n_slots: int = 8, compact_overflow: bool = True):
        super().__init__()
        self.compact_overflow = bool(compact_overflow)
        self.d_k = d_k
        self.d_v = d_v
        self.n_slots = max(1, int(n_slots))
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        # Global prior still updated by classic replay (compat / long-term)
        self.M_s = nn.Parameter(torch.zeros(d_k, d_v))
        # Softmax temperature for slot attention (learnable log-scale)
        self.log_temp = nn.Parameter(torch.tensor(0.0))
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
        """Ring write: one segment of context into the next slot."""
        B, S, _, _ = slots.shape
        if M.dim() != 3 or M.size(0) != B:
            return slots, ptr, filled
        if weight is not None:
            # per-sequence strength; NOT normalised by the batch mean (that made B=1
            # serving always weight 1.0 while B>1 training did not)
            w = weight.detach().reshape(B, 1, 1).clamp(min=0.0)
            snap = M.detach() * w
        else:
            snap = M.detach()
        new_slots = slots.clone()
        new_filled = filled.clone()
        if self.compact_overflow and S > 1:
            # Tensor-only path (no bool(...any()) — avoids torch.compile graph breaks)
            full = filled[:, ptr] > 0  # (B,)
            nxt = (ptr + 1) % S
            c_old, c_nxt = filled[:, ptr], filled[:, nxt]
            tot = (c_old + c_nxt).clamp(min=1e-8)
            merged = (
                c_old.view(B, 1, 1) * slots[:, ptr] + c_nxt.view(B, 1, 1) * slots[:, nxt]
            ) / tot.view(B, 1, 1)
            new_slots[:, nxt] = torch.where(full.view(B, 1, 1), merged, slots[:, nxt])
            new_filled[:, nxt] = torch.where(full, c_old + c_nxt, c_nxt)
        new_slots[:, ptr] = snap
        new_filled[:, ptr] = 1.0
        new_ptr = (ptr + 1) % S
        return new_slots, new_ptr, new_filled

    def _slot_keys(self, slots: Tensor) -> Tensor:
        """(B, S, d_k) key summary per slot from the stored matrix."""
        # Row energy of M as a key fingerprint in R^{d_k}
        k = slots.norm(dim=-1)  # (B, S, d_k)
        return F.normalize(k, dim=-1, eps=1e-8)

    def read(
        self,
        d_t: Tensor,
        slots: Tensor | None = None,
        filled: Tensor | None = None,
    ) -> Tensor:
        """Query all filled context slots + global prior."""
        q = self.W_q(d_t)
        q = F.normalize(q, dim=-1, eps=1e-8)
        u_prior = torch.einsum("kd,bk->bd", self.M_s, q)

        if slots is None or filled is None:
            return u_prior

        # Per-slot value read: (B, S, d_v)
        r = torch.einsum("bskd,bk->bsd", slots, q)
        # Content address over slots
        k_slot = self._slot_keys(slots)
        scores = torch.einsum("bk,bsk->bs", q, k_slot)
        temp = self.log_temp.exp().clamp(min=0.1, max=10.0)
        scores = scores / temp
        scores = scores.masked_fill(filled <= 0, -1e9)

        any_fill = (filled.sum(dim=-1, keepdim=True) > 0).float()  # (B, 1)
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha * any_fill  # zero mass if nothing filled
        u_ctx = torch.einsum("bs,bsd->bd", alpha, r)

        # Prefer context ring when slots exist; keep a small prior residual
        return any_fill * (0.9 * u_ctx + 0.1 * u_prior) + (1.0 - any_fill) * u_prior

    def read_seq(
        self,
        D: Tensor,  # (B, T, d_model)
        slots: Tensor | None = None,  # (B, S, d_k, d_v)
        filled: Tensor | None = None,  # (B, S)
    ) -> Tensor:
        """Token-parallel read: same ring for all t, no B*T slot expand."""
        B, T, _ = D.shape
        q = F.normalize(self.W_q(D), dim=-1, eps=1e-8)  # (B, T, d_k)
        u_prior = torch.einsum("btk,kd->btd", q, self.M_s)

        if slots is None or filled is None:
            return u_prior

        # (B, T, S, d_v)
        r = torch.einsum("bskd,btk->btsd", slots, q)
        k_slot = self._slot_keys(slots)  # (B, S, d_k)
        scores = torch.einsum("btk,bsk->bts", q, k_slot)
        temp = self.log_temp.exp().clamp(min=0.1, max=10.0)
        scores = scores / temp
        scores = scores.masked_fill((filled <= 0).unsqueeze(1), -1e9)
        any_fill = (filled.sum(dim=-1) > 0).float().view(B, 1, 1)
        alpha = torch.softmax(scores, dim=-1) * any_fill
        u_ctx = torch.einsum("bts,btsd->btd", alpha, r)
        return any_fill * (0.9 * u_ctx + 0.1 * u_prior) + (1.0 - any_fill) * u_prior

    def replay(self, M: Tensor, surprises: Tensor, eta: float) -> None:
        """Legacy global prior update (batch-level consolidation)."""
        if M.dim() != 3:
            return
        B = M.size(0)
        s = surprises.detach().reshape(-1).clamp(min=0.0)
        if s.numel() != B:
            s = s.mean().expand(B)
        num = (s.view(B, 1, 1) * M.detach()).sum(0)
        den = s.sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # data-parallel: every rank must apply the SAME update or M_s (not a DDP-synced grad) diverges
            torch.distributed.all_reduce(num)
            torch.distributed.all_reduce(den)
        avg = num / (den + 1e-8)
        self.M_s.data.mul_(1.0 - eta).add_(eta * avg)

    def tie_query_to(self, W_k: nn.Linear) -> None:
        """Align slow query basis with fast key basis (context retrieval match)."""
        with torch.no_grad():
            self.W_q.weight.copy_(W_k.weight)
