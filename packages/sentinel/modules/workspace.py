"""Learned Global Workspace (J-space) for SENTINEL / AERIS.

Why this module exists
----------------------
The first J-space (``Metacognition._update_jspace``) is a rule-based slot bank:
random slot vectors, written from a few scalar signals, kept as *module* state.
That has four problems (all measured, see ``tests/test_workspace.py``):

  1. Not learned    - the broadcast is detached; only ``W_js`` / ``ws_bias`` train.
  2. Not per-token  - one broadcast vector for a whole chunk, computed from the
                      PREVIOUS call's state (in training that state is ``None``,
                      so the "content" is just 4 scalars).
  3. Leaks          - slots live on the Metacognition object, so the same input
                      gives different logits on a second run / after another user.
  4. Costs syncs    - ~8 ``.item()`` host syncs per forward (every decoded token).

Design (follows Gurnee et al. 2026, "Verbalizable Representations Form a Global
Workspace in Language Models")
-----------------------------------------------------------------------------
    lens(h)   = softmax( W_U . norm( J h ) )          # J-lens readout
    J-space   = sparse, NON-NEGATIVE combination of lens vectors (k <= ~25)
    broadcast = a write-back of that sparse code to every consumer

  * ``J``  = I + B A (rank r, B zero-init)  ->  at init J = I, which is exactly the
             logit lens (the paper's special case); training can refine it.
  * lens vectors = ``atoms`` (n_atoms, d_model): a learned dictionary.
  * limited capacity = top-k over ReLU scores (only ``k_active`` atoms ignite).
  * broadcast = ``V(code)`` -> (d_ws), consumed by NeuralField.W_js and
             SkillRouter.ws_bias (both zero-init, so enabling it is exactly
             neutral at step 0).
  * reportable = ``read_tokens`` / ``atom_words`` project onto the unembedding.

The module is a pure function of the field state at each position
(read -> compete -> broadcast), so it is causal, per-sequence, deterministic and
keeps full-sequence == token-by-token decode parity. Persistence across tokens
comes from the recurrent state H / fast memory, not from hidden module state.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GlobalWorkspace(nn.Module):
    """Sparse, learned, capacity-limited workspace read from the field state."""

    def __init__(
        self,
        d_model: int,
        d_ws: int,
        n_atoms: int = 128,
        k_active: int = 8,
        rank: int = 16,
    ):
        super().__init__()
        if d_ws <= 0:
            raise ValueError("GlobalWorkspace needs d_ws > 0 (cfg.jspace_dim)")
        self.d_model = int(d_model)
        self.d_ws = int(d_ws)
        self.n_atoms = max(2, int(n_atoms))
        # Paper: ~20-25 concepts are meaningfully active at once. Keep it limited.
        self.k_active = max(1, min(int(k_active), self.n_atoms))
        r = max(1, min(int(rank), self.d_model))

        # J = I + B A   (B zero-init => J = I => plain logit lens at init)
        self.lens_in = nn.Linear(self.d_model, r, bias=False)
        self.lens_out = nn.Linear(r, self.d_model, bias=False)
        nn.init.zeros_(self.lens_out.weight)
        self.norm = nn.LayerNorm(self.d_model)

        # Dictionary of lens vectors. norm(.) has ~unit variance per dim, so atoms
        # with entries ~N(0, 1/d) give O(1) scores.
        self.atoms = nn.Parameter(torch.randn(self.n_atoms, self.d_model) / math.sqrt(self.d_model))
        # Write-back (broadcast) of the sparse code into the consumers' space.
        self.broadcast = nn.Linear(self.n_atoms, self.d_ws, bias=False)

    # ------------------------------------------------------------------ core
    def lens_state(self, h: Tensor) -> Tensor:
        """norm(J h): the coordinates in which the unembedding reads the state."""
        return self.norm(h + self.lens_out(self.lens_in(h)))

    def code(self, h: Tensor, steer: Optional[Tensor] = None) -> Tensor:
        """Sparse non-negative workspace code, shape (..., n_atoms)."""
        z = self.lens_state(h)
        s = F.relu(z @ self.atoms.t())
        if steer is not None:
            # Steering / injection experiments: add to the pre-competition scores.
            s = F.relu(s + steer.to(dtype=s.dtype, device=s.device))
        if self.k_active < self.n_atoms:
            kth = s.topk(self.k_active, dim=-1).values[..., -1:]
            s = s * (s >= kth).to(s.dtype)  # competition for limited capacity
        return s

    def forward(self, h: Tensor, steer: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Return ``(broadcast (..., d_ws), code (..., n_atoms))``."""
        c = self.code(h, steer=steer)
        return self.broadcast(c), c

    # ------------------------------------------------------------ diagnostics
    @staticmethod
    @torch.no_grad()
    def stats(code: Tensor) -> dict[str, float]:
        """Host-side stats. Call only when logging (does a device sync)."""
        active = (code > 0).float().sum(-1)
        return {
            "ws_active": float(active.mean().item()),
            "ws_capacity": float((active / code.size(-1)).mean().item()),
            "ws_mass": float(code.sum(-1).mean().item()),
        }

    # ------------------------------------------------------------ reportability
    @torch.no_grad()
    def read_tokens(self, h: Tensor, W_U: Tensor, topn: int = 10) -> tuple[Tensor, Tensor]:
        """J-lens readout: top-n (token ids, probs) of ``softmax(W_U norm(J h))``.

        ``W_U`` is the unembedding / tied embedding weight, shape (V, d_model).
        """
        logits = self.lens_state(h).float() @ W_U.float().t()
        probs = logits.softmax(-1)
        vals, ids = probs.topk(min(int(topn), probs.size(-1)), dim=-1)
        return ids, vals

    @torch.no_grad()
    def atom_words(self, W_U: Tensor, topn: int = 5) -> Tensor:
        """Which vocabulary tokens each atom pushes toward, (n_atoms, topn) ids."""
        logits = self.atoms.float() @ W_U.float().t()
        return logits.topk(min(int(topn), logits.size(-1)), dim=-1).indices
