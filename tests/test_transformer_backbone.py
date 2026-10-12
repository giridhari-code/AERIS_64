"""Decoder-only Transformer is the default AERIS backbone."""

from __future__ import annotations

import torch

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField


def test_forward_and_causal():
    cfg = ModelConfig(
        architecture="transformer",
        d_model=64, n_layers=2, n_heads=4,
        vocab_size=50, max_seq_len=64, dropout=0.0,
    )
    m = NeuroField(cfg, SafetyConfig(enable_audit=False)).eval()
    x = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        out = m(x)
        assert out.logits.shape == (2, 20, 50)
        assert out.metrics.get("path") == "transformer"
        x2 = x.clone()
        x2[:, 12:] = (x2[:, 12:] + 3) % 50
        out2 = m(x2)
        assert torch.allclose(out.logits[:, :12], out2.logits[:, :12], atol=1e-5)


def test_train_step():
    cfg = ModelConfig(
        architecture="transformer",
        d_model=64, n_layers=2, n_heads=4,
        vocab_size=50, max_seq_len=64, dropout=0.1,
    )
    m = NeuroField(cfg)
    x = torch.randint(0, 50, (4, 16))
    out = m(x, targets=x, label_smoothing=0.1)
    out.loss.backward()
