"""Paper Appendix A.2 Change 5: decay-consistent delta rule."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sentinel.modules.memory import FastMemory


def test_decay_consistent_write_matches_formula():
    torch.manual_seed(0)
    B, D, dk, dv = 2, 16, 8, 8
    fm = FastMemory(D, dk, dv)
    M = torch.randn(B, dk, dv)
    d_prev = torch.randn(B, D)
    o = torch.randn(B, D)
    g = torch.ones(B, 1)

    M_new = fm.write(M, d_prev, o, g)

    with torch.no_grad():
        k = fm.W_k(d_prev)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        v = fm.W_v(o)
        lam = fm.lambda_diag.view(1, -1, 1)
        M_dec = lam * M
        pred = torch.einsum("bkd,bk->bd", M_dec, k)
        delta = v - pred
        expected = M_dec + g.unsqueeze(-1) * torch.einsum("bk,bd->bkd", k, delta)

    assert torch.allclose(M_new, expected, atol=1e-5)


def test_zero_gate_only_decays():
    """Proposition 4: g=0 → M_t = Λ M_{t-1}."""
    torch.manual_seed(1)
    fm = FastMemory(16, 8, 8)
    M = torch.randn(1, 8, 8)
    d_prev = torch.randn(1, 16)
    o = torch.randn(1, 16)
    g = torch.zeros(1, 1)
    M_new = fm.write(M, d_prev, o, g)
    expected = fm.lambda_diag.view(1, -1, 1) * M
    assert torch.allclose(M_new, expected, atol=1e-5)
