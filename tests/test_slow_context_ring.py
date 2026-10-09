"""Slow memory context-window ring: write slots, query all filled."""

import torch
from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField
from neurofield.modules.memory import SlowMemory


def test_n_slots_auto():
    cfg = ModelConfig(d_model=32, d_k=8, d_v=8, n_skills=2, top_k=1, k_max=1,
                      vocab_size=20, max_seq_len=64, context_slots=0)
    m = NeuroField(cfg, SafetyConfig(truncate_write_window=16))
    assert m.slow_mem.n_slots == 64 // 16


def test_ring_write_and_read():
    sm = SlowMemory(d_model=16, d_k=8, d_v=8, n_slots=4)
    B = 2
    slots, ptr, filled = sm.init_slots(B, torch.device("cpu"))
    assert ptr == 0 and filled.sum() == 0

    for i in range(5):  # wrap once
        M = torch.randn(B, 8, 8)
        M[:, i % 8, :] = float(i + 1)  # distinctive
        slots, ptr, filled = sm.write_slot(slots, ptr, filled, M)
    assert filled.sum() == B * 4  # all slots filled after wrap
    assert ptr == 1  # 5 writes → ptr 5 % 4 = 1

    d = torch.randn(B, 16)
    u = sm.read(d, slots=slots, filled=filled)
    assert u.shape == (B, 8)
    assert torch.isfinite(u).all()


def test_forward_fills_slots():
    cfg = ModelConfig(d_model=32, d_k=8, d_v=8, n_skills=2, top_k=1, k_max=1,
                      vocab_size=30, max_seq_len=64, dendrite_window=2, context_slots=0)
    m = NeuroField(cfg, SafetyConfig(truncate_write_window=8, enable_audit=False))
    x = torch.randint(0, 30, (1, 24))
    out = m(x)
    filled = out.state["slow_filled"]
    # segment ends at 8,16 + EOS → at least 3 writes
    assert float(filled.sum()) >= 3
    assert out.state["slow_slots"].shape[1] == m.slow_mem.n_slots
