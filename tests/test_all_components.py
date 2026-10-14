"""Every architectural component: shape, grad, paper-critical behaviour."""

from __future__ import annotations

import torch
import pytest

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig
from neurofield.model import SENTINEL
from neurofield.modules.dendrite import Dendrite
from neurofield.modules.field import NeuralField
from neurofield.modules.memory import FastMemory, SlowMemory
from neurofield.modules.metacognition import Metacognition
from neurofield.modules.neuromodulator import Neuromodulator
from neurofield.modules.predictor import Predictor
from neurofield.modules.router import SkillRouter


def test_dendrite():
    d = Dendrite(32, window=3)
    y = d(torch.randn(2, 3, 32), torch.randn(2, 32))
    assert y.shape == (2, 32) and torch.isfinite(y).all()


def test_field_R_and_grad():
    f = NeuralField(32, 16)
    assert f.R.weight.shape == (1, 1, 3)
    h = torch.randn(2, 32, requires_grad=True)
    out = f(h, torch.randn(2, 32), torch.randn(2, 16), k=2)
    out.sum().backward()
    assert h.grad is not None and float(h.grad.detach().norm()) > 0


def test_predictor_surprise_detached():
    pred = Predictor(32)(torch.randn(2, 32))
    e, S = Predictor.compute_error_and_surprise(pred, torch.randn(2, 32))
    assert e.shape == (2, 32) and S.shape == (2,) and not S.requires_grad


def test_fast_memory_write_grads():
    fm = FastMemory(32, 16, 16)
    d = torch.randn(2, 32, requires_grad=True)
    o = torch.randn(2, 32, requires_grad=True)
    g = torch.ones(2, 1, requires_grad=True)
    M = fm.write(torch.zeros(2, 16, 16), d, o, g)
    M.sum().backward()
    assert d.grad is not None and o.grad is not None and g.grad is not None


def test_slow_ring_and_replay():
    sm = SlowMemory(32, 16, 16, n_slots=4)
    slots, ptr, filled = sm.init_slots(1, torch.device("cpu"))
    for _ in range(5):
        slots, ptr, filled = sm.write_slot(slots, ptr, filled, torch.randn(1, 16, 16))
    # ring full: oldest slot is folded into its neighbour, not erased (count-weighted), so all 5 writes are kept
    assert float((filled > 0).sum()) == 4.0 and float(filled.sum()) == 5.0
    before = sm.M_s.detach().clone()
    sm.replay(torch.randn(2, 16, 16), torch.tensor([1.0, 2.0]), 0.5)
    assert not torch.allclose(before, sm.M_s)


def test_neuromod_and_metacog():
    g = Neuromodulator(32)(torch.tensor([0.3, 1.8]), torch.randn(2, 32))
    assert g.shape == (2, 1)
    ctrl = Metacognition(k_max=4).step(
        S=torch.tensor([1.0, 2.0]), bar_S=None, router_entropy=torch.tensor([0.4, 1.2])
    )
    assert 1 <= ctrl.k <= 4 and ctrl.gate_bias.shape == (2,)


def test_router_ste():
    r = SkillRouter(32, 4, 2)
    h = torch.randn(2, 32, requires_grad=True)
    o, pi, w = r(h)
    o.sum().backward()
    assert r.router.weight.grad is not None
    assert torch.allclose(pi.sum(-1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize(
    "ab",
    [
        AblationConfig(),
        AblationConfig(use_fast_memory=False),
        AblationConfig(use_slow_memory=False),
        AblationConfig(use_neuromodulator=False),
        AblationConfig(use_metacognition=False),
    ],
)
def test_full_model_and_ablations(ab):
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=2,
        vocab_size=40, max_seq_len=64, dendrite_window=2, dropout=0.0,
    )
    m = SENTINEL(cfg, SafetyConfig(truncate_write_window=8, enable_audit=True))
    m.set_ablation(ab)
    out = m(torch.randint(0, 40, (2, 16)), targets=torch.randint(0, 40, (2, 16)), return_audit=True)
    assert out.logits.shape == (2, 16, 40)
    assert out.loss is not None and torch.isfinite(out.loss)
