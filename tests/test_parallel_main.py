"""Main path is Parallel Form (sequential time-loop removed)."""

from __future__ import annotations

import torch

from neurofield.config import ModelConfig, SafetyConfig, AblationConfig
from neurofield.model import SENTINEL


def _make():
    cfg = ModelConfig(
        d_model=48, d_k=24, d_v=24, n_skills=2, top_k=1, k_max=2,
        vocab_size=32, max_seq_len=64, dendrite_window=2, dropout=0.0,
    )
    return SENTINEL(cfg, SafetyConfig(enable_audit=False, truncate_write_window=16))


def test_forward_shape_and_path():
    m = _make()
    x = torch.randint(0, 32, (2, 12))
    out = m(x)
    assert out.logits.shape == (2, 12, 32)
    assert out.metrics.get("path") == "parallel"


def test_forward_loss_backward():
    m = _make()
    x = torch.randint(0, 32, (2, 12))
    y = torch.randint(0, 32, (2, 12))
    out = m(x, targets=y)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()


def test_leakage():
    m = _make().eval()
    x = torch.randint(2, 30, (2, 16))
    with torch.no_grad():
        a = m(x).logits
        x2 = x.clone()
        x2[:, 7] = (x2[:, 7] + 5) % 30
        b = m(x2).logits
        diff = (a[:, :7] - b[:, :7]).abs().max().item()
    assert diff < 1e-5, diff


def test_ablation_no_fast():
    m = _make()
    m.set_ablation(AblationConfig(use_fast_memory=False))
    out = m(torch.randint(0, 32, (2, 10)))
    assert out.logits.shape == (2, 10, 32)
