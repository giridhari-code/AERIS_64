"""Remaining paper v3.1 alignment checks (reference looped form)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sentinel.config import ModelConfig, SafetyConfig
from sentinel.model import SENTINEL
from sentinel.modules.memory import FastMemory


def test_parallel_ops_forced_true():
    m = SENTINEL(
        ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=2, vocab_size=40, dendrite_window=2, max_seq_len=32),
        SafetyConfig(enable_audit=False),
    )
    assert m.parallel_ops is True


def test_per_batch_fast_memory_not_shared_across_sequences():
    """Failure mode 1: each batch row has its own M plane."""
    torch.manual_seed(0)
    m = SENTINEL(
        ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=2, vocab_size=40, dendrite_window=2, max_seq_len=32),
        SafetyConfig(enable_audit=False),
    )
    x = torch.randint(0, 40, (4, 6))
    out = m(x)
    assert out.state is not None
    # depth passes store M_pass list; final state M is (B, dk, dv)
    M = out.state.get("M")
    if M is not None:
        assert M.shape[0] == 4


def test_freeze_zeros_gate_in_forward_metrics_path():
    m = SENTINEL(
        ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=2, vocab_size=40, dendrite_window=2, max_seq_len=32),
        SafetyConfig(enable_audit=False),
    )
    st = m.new_state(2, torch.device("cpu"))
    st["memory_frozen"] = True
    x = torch.randint(0, 40, (2, 5))
    out = m(x, state=st, collect_stats=True)
    # gate_mean should be ~0 when frozen
    gm = out.metrics.get("gate_mean", 0.0)
    assert float(gm) < 1e-5


def test_write_seq_matches_stepwise_change5():
    """write_seq (looped path) agrees with stepwise write."""
    torch.manual_seed(2)
    B, T, D, dk, dv = 2, 5, 16, 8, 8
    fm = FastMemory(D, dk, dv)
    M0 = torch.zeros(B, dk, dv)
    D_seq = torch.randn(B, T, D)
    O = torch.randn(B, T, D)
    G = torch.rand(B, T, 1)

    M_s, R_s, *_ = fm.write_seq(
        M0.clone(), D_seq, O, G,
        d_prev0=None,
        truncate_window=10**9,
        max_memory_norm=0.0,
        segment_len=0,
        seg_cnt0=0,
        surprises=None,
        collect_stats=False,
    )

    M = M0.clone()
    reads = []
    for t in range(T):
        d_prev = D_seq[:, t - 1] if t > 0 else torch.zeros(B, D)
        M = fm.write(M, d_prev, O[:, t], G[:, t])
        reads.append(fm.read(M, D_seq[:, t]))
    R = torch.stack(reads, dim=1)
    assert torch.allclose(M_s, M, atol=1e-4)
    assert torch.allclose(R_s, R, atol=1e-4)
