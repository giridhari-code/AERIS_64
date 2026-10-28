"""Regression tests for inference-engine bugs (self-heal of the real carry, stale slot keys, batch EOS)."""
from __future__ import annotations

import torch

from sentinel.config import AblationConfig, ModelConfig
from sentinel.model import SENTINEL
from sentinel.modules.inference_cache import InferenceCache


def _model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=32, d_k=8, d_v=8, n_skills=4, top_k=2, k_max=2,
        dendrite_window=3, vocab_size=40, max_seq_len=64,
    )
    m = SENTINEL(cfg, ablation=AblationConfig(use_metacognition=False, fixed_k=2))
    m.eval()
    return m


def test_heal_repairs_per_pass_carry():
    m = _model()
    with torch.no_grad():
        st = m(torch.randint(0, 40, (1, 6))).state
        st["M_pass"][0][0, 0, 0] = float("nan")
        st["Hin"][0][0, 0] = float("inf")
        st["D_pass"][0][0, 0] = float("nan")
        healed = m.heal_session_state(st)
        for key in ("M_pass", "Hin", "D_pass", "R_pass"):
            assert all(torch.isfinite(x).all() for x in healed[key]), key
        out = m(torch.randint(0, 40, (1, 1)), state=healed)
        assert torch.isfinite(out.logits).all()


def test_heal_drops_stale_slot_keys():
    m = _model()
    cache = InferenceCache.from_model(m, batch=1)
    cache.prefill(m, torch.randint(0, 40, (1, 8)))
    assert "slow_slot_keys" in cache.state
    healed = m.heal_session_state(cache.state)
    assert "slow_slot_keys" not in healed
    # consolidate_turn writes the ring -> keys must not survive either
    cache.update_slot_keys(m)
    cons = m.consolidate_turn(cache.state)
    assert "slow_slot_keys" not in cons


def test_cached_decode_matches_uncached_logits():
    m = _model()
    ids = torch.randint(0, 40, (1, 12))
    with torch.no_grad():
        full = m(ids).logits
        cache = InferenceCache.from_model(m, batch=1)
        cache.prefill(m, ids[:, :6])
        steps = [cache.decode_step(m, ids[:, t:t + 1])[:, -1] for t in range(6, 12)]
    got = torch.stack(steps, dim=1)
    assert torch.allclose(got, full[:, 6:], atol=1e-3), (got - full[:, 6:]).abs().max().item()


def test_batch_generate_freezes_rows_after_eos():
    m = _model()
    eos = 3
    ids = torch.randint(4, 40, (2, 5))
    out = InferenceCache.from_model(m, batch=2).generate(m, ids, max_new_tokens=12, temperature=0, eos_id=eos)
    for row in out[:, 5:]:
        pos = (row == eos).nonzero()
        if len(pos):
            assert bool((row[pos[0, 0]:] == eos).all())


def test_write_seq_t1_fast_path_matches_prefix_scan():
    from sentinel.modules.memory import FastMemory

    torch.manual_seed(1)
    B, T, D, dk, dv = 2, 12, 16, 8, 8
    fm = FastMemory(D, dk, dv)
    D_seq, O = torch.randn(B, T, D), torch.randn(B, T, D)
    G = torch.sigmoid(torch.randn(B, T, 1))
    M_full, R_full, *_ = fm.write_seq(torch.zeros(B, dk, dv), D_seq, O, G)

    M, d_prev, reads = torch.zeros(B, dk, dv), None, []
    for t in range(T):  # T=1 calls take the rank-1 fast path
        M, R_t, *_ = fm.write_seq(M, D_seq[:, t:t + 1], O[:, t:t + 1], G[:, t:t + 1], d_prev0=d_prev)
        reads.append(R_t)
        d_prev = D_seq[:, t]
    assert torch.allclose(M, M_full, atol=1e-4), (M - M_full).abs().max().item()
    assert torch.allclose(torch.cat(reads, 1), R_full, atol=1e-4)


def test_skip_meta_in_eval_keeps_logits():
    m = _model()
    m.ablation.use_metacognition = True
    ids = torch.randint(0, 40, (1, 10))

    def run():
        c = InferenceCache.from_model(m, batch=1)
        outs = [c.prefill(m, ids[:, :5])[:, -1]]
        outs += [c.decode_step(m, ids[:, t:t + 1])[:, -1] for t in range(5, 10)]
        return torch.stack(outs)

    with torch.no_grad():
        m.cfg.skip_meta_in_eval = False
        ref = run()
        m.cfg.skip_meta_in_eval = True
        fast = run()
    assert torch.allclose(ref, fast, atol=1e-5), (ref - fast).abs().max().item()
