"""Hierarchical multi-timescale prediction error tests."""

from __future__ import annotations

import torch

from sentinel.config import ModelConfig
from sentinel.model import SENTINEL
from sentinel.modules.predictor import HierarchicalPredictor, Predictor


def test_future_mean_shape():
    X = torch.randn(2, 20, 32)
    y = HierarchicalPredictor._future_mean(X, 4)
    assert y.shape == X.shape
    # last positions pad with last token — finite
    assert torch.isfinite(y).all()


def test_level_errors_keys():
    hp = HierarchicalPredictor(32, mid_horizon=4, coarse_horizon=8)
    H = torch.randn(2, 16, 32)
    X = torch.randn(2, 16, 32)
    le = hp.level_errors(H, X)
    assert "err_fine" in le and "err_mid" in le and "err_coarse" in le
    assert le["err_mid"].ndim == 0
    assert le["err_coarse"].item() >= 0


def test_forward_api_compatible():
    hp = HierarchicalPredictor(32)
    h = torch.randn(4, 32)
    y = hp(h)
    assert y.shape == h.shape


def test_model_train_loss_has_levels():
    cfg = ModelConfig(
        d_model=64,
        d_k=32,
        d_v=32,
        n_skills=4,
        top_k=2,
        k_max=2,
        dendrite_window=3,
        vocab_size=128,
        max_seq_len=64,
        hierarchical_predict=True,
        pred_mid_horizon=4,
        pred_coarse_horizon=8,
        lambda_err_mid=0.05,
        lambda_err_coarse=0.02,
    )
    m = SENTINEL(cfg)
    m.train()
    x = torch.randint(0, 128, (2, 24))
    out = m(x, targets=x)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert "err_mid" in out.metrics
    assert "err_coarse" in out.metrics
    # backward
    out.loss.backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in m.predictor.parameters()
        if p.requires_grad
    )
    assert grad_ok


def test_eval_unchanged_logits_finite():
    cfg = ModelConfig(
        d_model=64, d_k=32, d_v=32, n_skills=4, top_k=2, k_max=2,
        dendrite_window=3, vocab_size=128, max_seq_len=64,
    )
    m = SENTINEL(cfg).eval()
    x = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        out = m(x)
    assert out.logits.shape == (1, 12, 128)
    assert torch.isfinite(out.logits).all()
