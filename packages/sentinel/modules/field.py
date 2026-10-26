"""Looped / recurrent-depth Neural Field for SENTINEL.

Parameter-shared block applied ``k`` times (Astra-style recurrent depth).
Metacognition selects ``k``; optional J-space broadcast biases each step.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class NeuralField(nn.Module):
    """Shared-weight recurrent field. Effective depth = k tied applications."""

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        *,
        reinject_input: bool = True,
        loop_residual_scale: bool = True,
        jspace_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.reinject_input = bool(reinject_input)
        self.loop_residual_scale = bool(loop_residual_scale)

        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.beta = nn.Parameter(torch.ones(d_model) * 0.1)
        self.zeta = nn.Parameter(torch.ones(d_model) * 0.5)
        self.W_m = nn.Linear(d_mem, d_model, bias=False)

        self.R = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.R.weight)
        self.R.weight.data[0, 0, 1] = 1.0

        self.norm = nn.LayerNorm(d_model)

        self.jspace_dim = int(jspace_dim)
        if self.jspace_dim > 0:
            self.W_js = nn.Linear(self.jspace_dim, d_model, bias=False)
            nn.init.zeros_(self.W_js.weight)
        else:
            self.W_js = None

        self.loop_damp = nn.Parameter(torch.tensor(0.15))

    def project_js(self, js: Optional[Tensor], batch: int) -> Optional[Tensor]:
        """Project workspace once; reuse across all k loops (call outside the loop)."""
        if js is None or self.W_js is None:
            return None
        if js.ndim == 1:
            js = js.unsqueeze(0).expand(batch, -1)
        elif js.size(0) == 1 and batch > 1:
            js = js.expand(batch, -1)
        if js.size(-1) != self.jspace_dim:
            if js.size(-1) > self.jspace_dim:
                js = js[..., : self.jspace_dim]
            else:
                pad = js.new_zeros(js.size(0), self.jspace_dim)
                pad[:, : js.size(-1)] = js
                js = pad
        return self.W_js(js)

    def single_step(
        self,
        h: Tensor,
        d_t: Tensor,
        r_prev: Tensor,
        js: Optional[Tensor] = None,
        loop_idx: int = 0,
        total_k: int = 1,
        *,
        js_proj: Optional[Tensor] = None,
    ) -> Tensor:
        """One tied field update.

        Prefer passing ``js_proj`` (from ``project_js``) when calling in a k-loop
        so the workspace Linear is not re-run every step.
        """
        local = self.R(torch.tanh(h).unsqueeze(1)).squeeze(1)
        mem = self.W_m(r_prev)

        if js_proj is not None:
            drive = d_t + local + mem + js_proj
        elif js is not None and self.W_js is not None:
            proj = self.project_js(js, h.size(0))
            drive = d_t + local + mem + proj if proj is not None else d_t + local + mem
        else:
            drive = d_t + local + mem

        if self.reinject_input:
            # d_t already in drive; no extra term needed beyond design
            pass

        cand = self.norm(self.alpha * h + self.beta * drive)
        z = self.zeta
        if self.loop_residual_scale and total_k > 1:
            # later loops refine more gently
            damp = 1.0 / (1.0 + self.loop_damp.abs() * float(loop_idx))
            z = z * damp
        return (1.0 - z) * h + z * cand

    def forward(
        self,
        h: Tensor,
        d_t: Tensor,
        r_prev: Tensor,
        k: int = 1,
        js: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply shared block ``k`` times with a single workspace projection."""
        k = max(1, int(k))
        js_proj = self.project_js(js, h.size(0)) if js is not None else None
        for i in range(k):
            h = self.single_step(
                h, d_t, r_prev, js=None, loop_idx=i, total_k=k, js_proj=js_proj
            )
        return h
