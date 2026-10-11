"""Parallel Form: leakage + smoke (paper §3 evidence path)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField

V = 40


def _model():
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=4, top_k=2, k_max=2,
        vocab_size=V, dendrite_window=3, max_seq_len=64, dropout=0.0,
    )
    return NeuroField(cfg, SafetyConfig(enable_audit=False, truncate_write_window=16))


def test_parallel_no_label_leakage():
    torch.manual_seed(0)
    m = _model().eval()
    x = torch.randint(0, V, (2, 16))
    with torch.no_grad():
        base = m(x, parallel=True).logits
        for t in (0, 4, 10, 14):
            x2 = x.clone()
            x2[:, t + 1 :] = torch.randint(0, V, x2[:, t + 1 :].shape)
            alt = m(x2, parallel=True).logits
            assert torch.equal(base[:, : t + 1], alt[:, : t + 1]), f"leak at t={t}"


def test_parallel_forward_shapes_and_loss():
    m = _model()
    x = torch.randint(0, V, (2, 12))
    y = torch.randint(0, V, (2, 12))
    out = m(x, targets=y, parallel=True)
    assert out.logits.shape == (2, 12, V)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.metrics.get("path") == "parallel"


def test_sequential_still_default():
    """parallel=False must stay the default path."""
    m = _model()
    x = torch.randint(0, V, (1, 8))
    out = m(x)
    assert out.logits.shape == (1, 8, V)
    # sequential metrics should not force path=parallel
    assert out.metrics.get("path") != "parallel"


def test_parallel_recall_smoke_loss_drops():
    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=48, d_k=24, d_v=24, n_skills=2, top_k=1, k_max=2,
        vocab_size=20, max_seq_len=64, dendrite_window=2, dropout=0.0,
    )
    m = NeuroField(cfg, SafetyConfig(truncate_write_window=16, enable_audit=False))
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    losses = []
    m.train()
    for _ in range(30):
        # simple copy pattern
        x = torch.randint(0, 20, (4, 16))
        y = x.clone()
        opt.zero_grad()
        out = m(x, targets=y, parallel=True, lambda_err=0.02, lambda_bal=0.0)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
    assert losses[-1] < losses[0] * 0.98 or losses[-1] < losses[0] - 0.2
