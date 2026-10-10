"""Smoke tests for NeuroField v2 – no external data required."""

from __future__ import annotations

import torch

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField


def test_forward_shape_and_grad():
    cfg = ModelConfig(
        d_model=64,
        d_k=32,
        d_v=32,
        n_skills=4,
        top_k=2,
        k_max=3,
        vocab_size=100,
        dendrite_window=2,
        tie_embeddings=True,
    )
    model = NeuroField(cfg, SafetyConfig(enable_audit=True))
    model.train()

    B, T = 2, 16
    x = torch.randint(0, 100, (B, T))
    y = torch.randint(0, 100, (B, T))

    out = model(x, targets=y, return_audit=True)
    assert out.logits.shape == (B, T, 100)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert out.audit is not None
    assert "gates" in out.audit

    out.loss.backward()

    # All major modules must receive gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


def test_param_report():
    cfg = ModelConfig(d_model=32, vocab_size=50, n_skills=2)
    model = NeuroField(cfg)
    report = model.param_report()
    assert "total" in report
    assert report["total"] > 0


def test_freeze_writes():
    cfg = ModelConfig(d_model=32, vocab_size=50)
    model = NeuroField(cfg)
    model.freeze_writes()
    # Gate should collapse near zero
    h = torch.randn(2, 32)
    rel = torch.ones(2)
    g = model.neuromod(rel, h)
    assert (g < 0.01).all()


def _tiny(**safety):
    cfg = ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=3, top_k=2, k_max=2,
                      vocab_size=50, dendrite_window=2)
    return NeuroField(cfg, SafetyConfig(enable_audit=True, **safety)).eval()


def test_audit_is_plain_floats_one_per_step():
    m = _tiny()
    out = m(torch.randint(0, 50, (1, 7)), return_audit=True)
    for key in ("gates", "write_norms", "mem_norms", "router_entropy"):
        vals = out.audit[key]
        assert len(vals) == 7 and all(isinstance(v, float) for v in vals), key


def test_write_norm_cap_is_enforced_without_branching():
    m = _tiny(max_write_norm=0.5, max_memory_norm=0.0)
    out = m(torch.randint(0, 50, (1, 1)))
    assert out.state["M"].norm().item() <= 0.5 + 1e-4


def test_no_cross_session_leak_when_global_tracking_off():
    # With track_global=True the 2nd call would inherit the 1st call's surprise baseline.
    m = _tiny()
    m.meta.track_global = False
    x = torch.randint(0, 50, (1, 9))
    a = m(x).logits
    b = m(x).logits
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_truncate_window_comes_from_training_config():
    from neurofield.config import NeuroFieldConfig
    from neurofield.training import Trainer

    cfg = NeuroFieldConfig()
    cfg.training.truncate_write_window = 7
    cfg.training.device = "cpu"
    cfg.training.output_dir = "/tmp/nf_trainer_test"
    m = NeuroField(ModelConfig(d_model=16, d_k=8, d_v=8, n_skills=2, vocab_size=20))
    Trainer(m, cfg, train_loader=[])
    assert m._truncate_window == 7


if __name__ == "__main__":
    test_forward_shape_and_grad()
    test_param_report()
    test_freeze_writes()
    test_audit_is_plain_floats_one_per_step()
    test_write_norm_cap_is_enforced_without_branching()
    test_no_cross_session_leak_when_global_tracking_off()
    print("All smoke tests passed.")
