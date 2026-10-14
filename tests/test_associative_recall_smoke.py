"""Short smoke: fast memory path can fit a tiny 2-pair recall (not full paper 800-step claim)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import SENTINEL


def _batch(n_pairs: int = 2, n_keys: int = 8, n_vals: int = 8, batch: int = 4, device="cpu"):
    """Sequence: k1 v1 k2 v2 ... k1 ?  target=v1 (simplified probe)."""
    keys = torch.randint(1, n_keys + 1, (batch, n_pairs), device=device)
    vals = torch.randint(n_keys + 1, n_keys + n_vals + 1, (batch, n_pairs), device=device)
    # interleave
    seq_len = n_pairs * 2 + 2  # pairs + probe key + answer slot
    x = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    for b in range(batch):
        pos = 0
        for i in range(n_pairs):
            x[b, pos] = keys[b, i]
            y[b, pos] = keys[b, i]
            pos += 1
            x[b, pos] = vals[b, i]
            y[b, pos] = vals[b, i]
            pos += 1
        # probe first key, predict first val
        x[b, pos] = keys[b, 0]
        y[b, pos] = keys[b, 0]
        pos += 1
        x[b, pos] = keys[b, 0]  # placeholder
        y[b, pos] = vals[b, 0]
    return x, y


def test_field_R_convolves_over_neurons():
    from neurofield.modules.field import NeuralField

    f = NeuralField(16, 8)
    h = torch.randn(2, 16)
    d = torch.randn(2, 16)
    r = torch.randn(2, 8)
    out = f(h, d, r, k=2)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    # R weight is 1x1x3
    assert f.R.weight.shape == (1, 1, 3)


def test_recall_smoke_loss_drops():
    torch.manual_seed(0)
    V = 20
    cfg = ModelConfig(
        d_model=48, d_k=24, d_v=24, n_skills=2, top_k=1, k_max=2,
        vocab_size=V, max_seq_len=64, dendrite_window=2, dropout=0.0,
    )
    m = SENTINEL(cfg, SafetyConfig(truncate_write_window=16, enable_audit=False))
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    losses = []
    m.train()
    for step in range(40):
        x, y = _batch(device="cpu")
        # clamp ids into vocab
        x = x.clamp(0, V - 1)
        y = y.clamp(0, V - 1)
        opt.zero_grad()
        out = m(x, targets=y, lambda_err=0.02, lambda_bal=0.0)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
    assert losses[-1] < losses[0] * 0.95 or losses[-1] < losses[0] - 0.5
