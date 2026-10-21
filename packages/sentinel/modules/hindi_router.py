"""Hindi / Hinglish specialized Skill Router for Indo-Voice-3.

Extends the original SkillRouter idea with language-aware routing
and dedicated skills for common Indian conversational patterns.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sentinel.voice_config import ModelConfig, VoiceConfig


# Default skill names for Indo-Voice-3 (can be overridden via config later)
INDO_SKILL_NAMES = [
    "general",           # 0 – default reasoning
    "hindi_response",    # 1 – pure Hindi generation bias
    "hinglish",          # 2 – code-mixed Hinglish
    "greeting",          # 3 – namaste / formal / informal openings
    "clarification",     # 4 – “samajh nahi aaya”, ask again
    "empathy",           # 5 – soft / caring tone
    "task_execution",    # 6 – tool / action oriented
    "closing",           # 7 – polite end of turn
]


class HindiSkillRouter(nn.Module):
    """
    Language-aware skill router.

    - Soft routing over n_skills
    - Optional language bias (Hindi / Hinglish preference)
    - Top-k sparsification for efficiency (same spirit as original)
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        voice_cfg: VoiceConfig,
        skill_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.d_model = model_cfg.d_model
        self.n_skills = model_cfg.n_skills
        self.top_k = min(model_cfg.top_k, model_cfg.n_skills)
        self.enable_hinglish = voice_cfg.enable_hinglish
        self.language_hint_weight = voice_cfg.language_hint_weight

        self.skill_names = skill_names or INDO_SKILL_NAMES[: self.n_skills]
        if len(self.skill_names) < self.n_skills:
            self.skill_names += [f"skill_{i}" for i in range(len(self.skill_names), self.n_skills)]

        # Router network
        self.router = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.n_skills),
        )

        # Per-skill output projections (same as classic SkillRouter idea)
        self.skill_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model, bias=False)
            for _ in range(self.n_skills)
        ])

        # Language bias vector (learnable)
        # Index 0 = neutral, 1 = Hindi preference, 2 = Hinglish preference
        self.lang_bias = nn.Parameter(torch.zeros(3, self.n_skills))
        nn.init.normal_(self.lang_bias, std=0.02)

    def forward(
        self,
        h: Tensor,
        language_id: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            h: (B, D) or (B*T, D) hidden states
            language_id: optional (B,) or (B*T,) long tensor
                         0 = neutral, 1 = Hindi, 2 = Hinglish

        Returns:
            o:      (..., D) mixed skill output
            pi:     (..., n_skills) routing probabilities
            weights:(..., top_k) selected skill weights (for audit)
        """
        logits = self.router(h)  # (..., n_skills)

        # Optional language bias
        if language_id is not None and self.language_hint_weight > 0:
            # language_id can be scalar or tensor
            if language_id.dim() == 0:
                bias = self.lang_bias[language_id]
            else:
                bias = self.lang_bias[language_id]  # (..., n_skills)
            logits = logits + self.language_hint_weight * bias

        pi = F.softmax(logits, dim=-1)

        # Top-k sparsification
        if self.top_k < self.n_skills:
            topk_vals, topk_idx = torch.topk(pi, self.top_k, dim=-1)
            mask = torch.zeros_like(pi).scatter_(-1, topk_idx, 1.0)
            pi_sparse = pi * mask
            pi_sparse = pi_sparse / pi_sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            pi_sparse = pi
            topk_vals = pi
            topk_idx = torch.arange(self.n_skills, device=h.device).expand(pi.shape[:-1] + (-1,))

        # Weighted skill outputs
        # Efficient path: compute all projections then weight
        skill_outs = torch.stack([proj(h) for proj in self.skill_projs], dim=-2)  # (..., n_skills, D)
        o = (skill_outs * pi_sparse.unsqueeze(-1)).sum(dim=-2)

        return o, pi_sparse, topk_vals

    def get_skill_name(self, idx: int) -> str:
        return self.skill_names[idx] if 0 <= idx < len(self.skill_names) else f"skill_{idx}"

    def param_report(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"hindi_router_total": total}
