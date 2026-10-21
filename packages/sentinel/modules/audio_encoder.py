"""Pure convolutional + SENTINEL-inspired Audio Encoder for Indo-Voice-3.

No Transformer layers.

Design
------
1. Mel-spectrogram frontend (fixed filterbank)
2. Conv stack (time + frequency downsampling)
3. SENTINEL-style local recurrence:
   - Dendrite-like temporal window features
   - Lightweight field update (depthwise residual)
   - Optional fast memory write/read over frames
4. Final projection to encoder_dim

Fully trainable end-to-end with SENTINEL core.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sentinel.voice_config import AudioConfig


# ---------------------------------------------------------------------------
# Mel frontend (unchanged, no learnable params)
# ---------------------------------------------------------------------------

class MelSpectrogramFrontend(nn.Module):
    """Differentiable mel-spectrogram frontend."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.sample_rate = cfg.sample_rate
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.n_mels = cfg.n_mels

        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb, persistent=False)
        window = torch.hann_window(cfg.win_length)
        self.register_buffer("window", window, persistent=False)

    def _create_mel_filterbank(self) -> Tensor:
        n_freqs = self.n_fft // 2 + 1
        f_min, f_max = 0.0, self.sample_rate / 2.0
        mel_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()

        fb = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center == left:
                center += 1
            if right == center:
                right += 1
            for j in range(left, center):
                if 0 <= j < n_freqs:
                    fb[i, j] = (j - left) / max(1, center - left)
            for j in range(center, right):
                if 0 <= j < n_freqs:
                    fb[i, j] = (right - j) / max(1, right - center)
        return fb

    def forward(self, waveform: Tensor) -> Tensor:
        """waveform (B, T) → mel (B, n_mels, T_frames)"""
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = stft.abs().pow(2)
        mel = torch.matmul(self.mel_fb, power)
        return torch.log(mel.clamp(min=1e-5))


# ---------------------------------------------------------------------------
# Pure Conv frontend
# ---------------------------------------------------------------------------

class ConvFrontend(nn.Module):
    """Strided conv stack: mel → lower time/freq resolution → encoder channels."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        C = cfg.conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(),
        )
        flat_dim = C * (cfg.n_mels // 4)
        self.proj = nn.Linear(flat_dim, cfg.encoder_dim)
        self.norm = nn.LayerNorm(cfg.encoder_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, mel: Tensor) -> Tensor:
        """mel (B, n_mels, T) → (B, T', encoder_dim)"""
        x = mel.unsqueeze(1)
        x = self.conv(x)
        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * M)
        x = self.drop(self.norm(self.proj(x)))
        return x


# ---------------------------------------------------------------------------
# SENTINEL-inspired temporal block (no attention / no Transformer)
# ---------------------------------------------------------------------------

class DendriteWindow(nn.Module):
    """Local temporal context — same spirit as SENTINEL Dendrite."""

    def __init__(self, dim: int, window: int = 5):
        super().__init__()
        self.window = window
        self.proj = nn.Linear(dim * window, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, T, D) → windowed features (B, T, D)"""
        B, T, D = x.shape
        pad = x.new_zeros(B, self.window - 1, D)
        xp = torch.cat([pad, x], dim=1)
        windows = xp.unfold(1, self.window, 1)
        windows = windows.permute(0, 1, 3, 2).contiguous().view(B, T, -1)
        return self.norm(self.proj(windows))


class FieldUpdate(nn.Module):
    """Lightweight residual field step (depthwise + pointwise, no attention)."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h: Tensor, d: Tensor) -> Tensor:
        residual = h
        x = self.norm1(h + d)
        x = self.dw(x.transpose(1, 2)).transpose(1, 2)
        x = residual + x
        residual = x
        x = residual + self.pw(self.norm2(x))
        return x


class FastFrameMemory(nn.Module):
    """Tiny frame-level fast memory (delta-rule style, sequential over T)."""

    def __init__(self, dim: int, d_k: int = 32, d_v: int = 32):
        super().__init__()
        self.W_k = nn.Linear(dim, d_k, bias=False)
        self.W_v = nn.Linear(dim, d_v, bias=False)
        self.W_q = nn.Linear(dim, d_k, bias=False)
        self.out = nn.Linear(d_v, dim, bias=False)
        self.d_k = d_k
        self.d_v = d_v
        self.scale = d_k ** -0.5

    def forward(self, x: Tensor, gate: Optional[Tensor] = None) -> Tensor:
        """x (B, T, D) → read features (B, T, D). Causal."""
        B, T, D = x.shape
        K = self.W_k(x)   # (B, T, d_k)
        V = self.W_v(x)   # (B, T, d_v)
        Q = self.W_q(x)   # (B, T, d_k)

        M = x.new_zeros(B, self.d_k, self.d_v)  # (B, d_k, d_v)
        reads = []
        for t in range(T):
            q = Q[:, t]                          # (B, d_k)
            # read: r = M^T @ q  → (B, d_v)
            r = torch.einsum('bkd,bk->bd', M, q)
            reads.append(self.out(r))

            k = K[:, t]                          # (B, d_k)
            v = V[:, t]                          # (B, d_v)
            # delta-rule write: M += g * (k v^T - decay)
            if gate is None:
                g = 1.0
            else:
                g = gate[:, t].view(B, 1, 1)
            kv = torch.einsum('bk,bd->bkd', k, v)  # (B, d_k, d_v)
            decay = (k * k).sum(-1).view(B, 1, 1) * 0.01
            M = M + g * (kv - M * decay)

        return torch.stack(reads, dim=1)


# ---------------------------------------------------------------------------
# Full Audio Encoder
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    Pure convolutional + SENTINEL-inspired audio encoder.

    Pipeline:
        waveform
          → MelSpectrogram
          → ConvFrontend
          → DendriteWindow (local temporal)
          → FieldUpdate × N
          → FastFrameMemory (optional)
          → features
    """

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.encoder_dim

        self.frontend = MelSpectrogramFrontend(cfg)
        self.conv = ConvFrontend(cfg)

        self.dendrite = DendriteWindow(dim, window=5)
        self.field_layers = nn.ModuleList([
            FieldUpdate(dim, dropout=cfg.dropout)
            for _ in range(max(1, cfg.encoder_layers // 2))
        ])
        self.use_memory = True
        self.frame_mem = FastFrameMemory(dim, d_k=32, d_v=32)

        self.final_norm = nn.LayerNorm(dim)
        self.pos_scale = nn.Parameter(torch.ones(1, 1, dim) * 0.02)

    def forward(
        self,
        waveform: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveform: (B, T_audio) float32 [-1, 1]
            lengths:  (B,) samples (optional)

        Returns:
            features: (B, T_frames, encoder_dim)
            frame_lengths: (B,)
        """
        mel = self.frontend(waveform)
        x = self.conv(mel)

        T = x.size(1)
        pos = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1)
        x = x + self.pos_scale * torch.sin(pos / 10000.0)

        d = self.dendrite(x)
        h = x
        for layer in self.field_layers:
            h = layer(h, d)

        if self.use_memory:
            r = self.frame_mem(h)
            h = h + r

        h = self.final_norm(h)

        if lengths is not None:
            approx = ((lengths // self.cfg.hop_length) + 1) // 4
            frame_lengths = approx.clamp(min=1, max=T)
        else:
            frame_lengths = torch.full((h.size(0),), T, device=h.device, dtype=torch.long)

        return h, frame_lengths

    def param_report(self) -> dict[str, int]:
        return {"audio_encoder_total": sum(p.numel() for p in self.parameters())}
