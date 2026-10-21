"""Modality Projector: maps AudioEncoder features → SENTINEL d_model space.

Keeps the core SENTINEL architecture unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from sentinel.voice_config import AudioConfig, ModelConfig


class ModalityProjector(nn.Module):
    """
    Projects audio encoder outputs into the language model embedding space.

    Design:
    - Linear + LayerNorm + optional residual GELU MLP
    - Learnable modality type embedding (audio vs text)
    - Supports both full-sequence and streaming (single frame) use
    """

    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig):
        super().__init__()
        self.audio_dim = audio_cfg.encoder_dim
        self.model_dim = model_cfg.d_model

        self.proj = nn.Sequential(
            nn.Linear(self.audio_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )

        # Learnable modality token (helps router / metacognition distinguish audio)
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        nn.init.normal_(self.modality_emb, std=0.02)

        self.drop = nn.Dropout(audio_cfg.dropout)

    def forward(
        self,
        audio_features: Tensor,
        add_modality_token: bool = True,
    ) -> Tensor:
        """
        Args:
            audio_features: (B, T, audio_dim)
            add_modality_token: prepend a learnable <AUDIO> token

        Returns:
            projected: (B, T' , d_model)  where T' = T or T+1
        """
        x = self.proj(audio_features)  # (B, T, d_model)
        x = self.drop(x)

        if add_modality_token:
            B = x.size(0)
            token = self.modality_emb.expand(B, -1, -1)
            x = torch.cat([token, x], dim=1)

        return x

    def param_report(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"modality_projector_total": total}
