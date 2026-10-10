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


def _small_model(window=8):
    cfg = ModelConfig(d_model=32, d_k=8, d_v=8, n_skills=2, top_k=1, k_max=1,
                      vocab_size=30, max_seq_len=64, dendrite_window=2, context_slots=0)
    m = NeuroField(cfg, SafetyConfig(truncate_write_window=window, enable_audit=False))
    m.eval()
    return m


def test_decode_steps_do_not_flood_the_ring():
    """1-token decode calls used to write a slot EVERY token (T-1 == 0)."""
    m = _small_model(window=8)
    with torch.no_grad():
        out = m(torch.randint(0, 30, (1, 5)))          # prefill: one chunk-end snapshot
        state = out.state
        for _ in range(20):                             # 20 decode steps
            out = m(torch.randint(0, 30, (1, 1)), state=state)
            state = out.state
    # 1 (prefill end) + decode writes at steps 8 and 16
    assert float(state["slow_filled"].sum()) == 3


def test_single_call_and_token_by_token_write_same_number_of_slots():
    m = _small_model(window=8)
    x = torch.randint(0, 30, (1, 24))
    with torch.no_grad():
        one_call = m(x).state
        state = None
        for t in range(24):
            state = m(x[:, t : t + 1], state=state).state
    assert float(one_call["slow_filled"].sum()) == float(state["slow_filled"].sum()) == 3
    assert one_call["seg_pos"] == state["seg_pos"] == 0


def test_slot_weight_is_batch_independent():
    sm = SlowMemory(d_model=16, d_k=8, d_v=8, n_slots=2)
    M = torch.ones(2, 8, 8)
    w = torch.tensor([2.0, 0.5])
    s2, _, f2 = sm.write_slot(*sm.init_slots(2, torch.device("cpu")), M, weight=w)
    s1, _, f1 = sm.write_slot(*sm.init_slots(1, torch.device("cpu")), M[:1], weight=w[:1])
    assert torch.allclose(s2[0, 0], s1[0, 0])  # same sequence => same slot, whatever the batch
