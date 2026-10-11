"""Tests for the Parallel Form (forward_parallel)."""

from __future__ import annotations

import torch
import pytest

from neurofield.config import ModelConfig, SafetyConfig, AblationConfig
from neurofield.model import NeuroField


def _make_model(d_model=48, d_k=24, d_v=24, vocab=32):
    cfg = ModelConfig(
        d_model=d_model, d_k=d_k, d_v=d_v,
        n_skills=2, top_k=1, k_max=2,
        vocab_size=vocab, max_seq_len=64, dendrite_window=2, dropout=0.0,
    )
    return NeuroField(cfg, SafetyConfig(enable_audit=False, truncate_write_window=16))


def test_parallel_forward_shape():
    m = _make_model()
    x = torch.randint(0, 32, (2, 12))
    out = m.forward_parallel(x, k_depth=2)
    assert out.logits.shape == (2, 12, 32)
    assert out.loss is None


def test_parallel_forward_with_loss():
    m = _make_model()
    x = torch.randint(0, 32, (2, 12))
    y = torch.randint(0, 32, (2, 12))
    out = m.forward_parallel(x, targets=y, k_depth=2)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_parallel_leakage():
    """Changing x[t+1] must not change logits at positions <= t."""
    m = _make_model()
    m.eval()
    x = torch.randint(2, 30, (2, 16))
    with torch.no_grad():
        logits1 = m.forward_parallel(x, k_depth=2).logits
        x2 = x.clone()
        x2[:, 7] = (x2[:, 7] + 5) % 30
        logits2 = m.forward_parallel(x2, k_depth=2).logits
        diff = (logits1[:, :7] - logits2[:, :7]).abs().max().item()
    assert diff < 1e-5, f"Leakage detected: max diff={diff}"


def test_parallel_ablation_no_fast_mem():
    m = _make_model()
    m.set_ablation(AblationConfig(use_fast_memory=False))
    x = torch.randint(0, 32, (2, 10))
    out = m.forward_parallel(x, k_depth=1)
    assert out.logits.shape == (2, 10, 32)


def test_parallel_flag_on_forward():
    m = _make_model()
    x = torch.randint(0, 32, (1, 8))
    out = m(x, parallel=True, k_depth=2) if False else m.forward_parallel(x, k_depth=2)
    # parallel=True path
    out2 = m(x, parallel=True)
    assert out2.logits.shape == (1, 8, 32)
    assert out2.metrics.get("path") == "parallel"


def test_sequential_still_default():
    m = _make_model()
    x = torch.randint(0, 32, (1, 8))
    out = m(x)
    assert out.logits.shape == (1, 8, 32)
    assert out.metrics.get("path") != "parallel"
