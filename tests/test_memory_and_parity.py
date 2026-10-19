"""Regression: fast-memory write, slow-ring causality, train/serve parity."""
from __future__ import annotations

import torch
from sentinel.config import ModelConfig, AblationConfig
from sentinel.model import SENTINEL
from sentinel.modules.memory import FastMemory, SlowMemory


def test_fast_memory_write_seq_matches_stepwise():
    torch.manual_seed(0)
    B, T, D, dk, dv = 2, 12, 16, 8, 8
    fm = FastMemory(D, dk, dv)
    M0 = torch.zeros(B, dk, dv)
    D_seq = torch.randn(B, T, D)
    O = torch.randn(B, T, D)
    G = torch.sigmoid(torch.randn(B, T, 1))

    M_par, R_par, _, _, _, _ = fm.write_seq(
        M0.clone(), D_seq, O, G, truncate_window=0, max_memory_norm=0.0
    )

    M = M0.clone()
    reads = []
    d_prev = torch.zeros(B, D)
    for t in range(T):
        M = fm.write(M, d_prev, O[:, t], G[:, t])
        reads.append(fm.read(M, D_seq[:, t]))
        d_prev = D_seq[:, t]
    R_seq = torch.stack(reads, dim=1)

    assert torch.allclose(R_par, R_seq, atol=1e-4), (
        f"read maxdiff {(R_par - R_seq).abs().max().item()}"
    )
    assert torch.allclose(M_par, M, atol=1e-4), (
        f"M maxdiff {(M_par - M).abs().max().item()}"
    )


def test_slow_ring_compact_preserves_mass():
    sm = SlowMemory(16, 8, 8, n_slots=4, compact_overflow=True)
    B = 1
    slots, ptr, filled = sm.init_slots(B, torch.device("cpu"))
    total_writes = 0
    for i in range(10):
        M = torch.randn(B, 8, 8)
        slots, ptr, filled = sm.write_slot(slots, ptr, filled, M)
        total_writes += 1
    # filled.sum tracks number of snapshots represented
    assert abs(float(filled.sum()) - total_writes) < 1e-4, filled.sum().item()


def test_model_parity_no_meta():
    torch.manual_seed(7)
    cfg = ModelConfig(
        d_model=32, d_k=8, d_v=8, n_skills=4, top_k=2, k_max=2,
        dendrite_window=3, vocab_size=40, max_seq_len=64,
    )
    model = SENTINEL(cfg, ablation=AblationConfig(use_metacognition=False, fixed_k=2))
    model.eval()
    ids = torch.randint(0, 40, (2, 10))
    with torch.no_grad():
        full = model(ids)
        state = None
        parts = []
        for t in range(ids.size(1)):
            o = model(ids[:, t : t + 1], state=state)
            parts.append(o.logits)
            state = o.state
        tok = torch.cat(parts, dim=1)
    diff = (full.logits - tok).abs().max().item()
    assert diff < 1e-4, f"parity diff {diff}"


def test_empty_sequence():
    cfg = ModelConfig(d_model=16, d_k=4, d_v=4, n_skills=2, top_k=1, k_max=1,
                      dendrite_window=2, vocab_size=20, max_seq_len=32)
    model = SENTINEL(cfg)
    out = model(torch.zeros(2, 0, dtype=torch.long))
    assert out.logits.shape == (2, 0, 20)


if __name__ == "__main__":
    test_fast_memory_write_seq_matches_stepwise()
    print("fast mem ok")
    test_slow_ring_compact_preserves_mass()
    print("slow ring ok")
    test_model_parity_no_meta()
    print("parity ok")
    test_empty_sequence()
    print("empty ok")
    print("all memory/parity tests passed")
