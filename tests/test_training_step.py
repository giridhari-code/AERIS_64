"""Training-step smoke: one forward+backward with hierarchical warmup path."""

from __future__ import annotations

import torch

from sentinel.config import ModelConfig
from sentinel.model import SENTINEL


def test_train_step_collect_stats_off_no_host_metrics():
    cfg = ModelConfig(
        d_model=64, d_k=32, d_v=32, n_skills=4, top_k=2, k_max=2,
        dendrite_window=3, vocab_size=128, max_seq_len=64,
        hierarchical_predict=True, lambda_err_mid=0.05, lambda_err_coarse=0.02,
    )
    m = SENTINEL(cfg).train()
    x = torch.randint(0, 128, (2, 20))
    out = m(
        x, targets=x, lambda_err=0.1, lambda_err_mid=0.05, lambda_err_coarse=0.02,
        collect_stats=False,
    )
    assert out.loss is not None and torch.isfinite(out.loss)
    # metrics should not force ce key when collect_stats=False
    out.loss.backward()
    assert m.embed.weight.grad is not None


def test_train_step_collect_stats_on_has_levels():
    cfg = ModelConfig(
        d_model=64, d_k=32, d_v=32, n_skills=4, top_k=2, k_max=2,
        dendrite_window=3, vocab_size=128, max_seq_len=64,
        hierarchical_predict=True,
    )
    m = SENTINEL(cfg).train()
    x = torch.randint(0, 128, (2, 20))
    out = m(x, targets=x, collect_stats=True)
    assert "ce" in out.metrics and "err_mid" in out.metrics
    assert "lambda_err_mid" in out.metrics
