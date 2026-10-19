"""Regression: parallel dendrite window must match sequential (causal past only)."""
from __future__ import annotations

import torch
from sentinel.modules.dendrite import Dendrite


def test_window_features_matches_sequential():
    torch.manual_seed(0)
    B, T, D, W = 2, 7, 12, 3
    dend = Dendrite(D, window=W)

    x_hist = torch.randn(B, W, D)
    X = torch.randn(B, T, D)
    Xp = torch.cat([x_hist, X], dim=1)

    D_par = dend.window_features(Xp, T)

    hist = x_hist.clone()
    seq = []
    for t in range(T):
        seq.append(dend.forward(hist, error=None))
        hist = torch.cat([hist[:, 1:], X[:, t : t + 1]], dim=1)
    D_seq = torch.stack(seq, dim=1)

    assert D_par.shape == D_seq.shape
    assert torch.allclose(D_par, D_seq, atol=1e-6), (
        f"max diff {(D_par - D_seq).abs().max().item()}"
    )


def test_decode_T1_matches_forward():
    torch.manual_seed(1)
    B, D, W = 4, 16, 3
    dend = Dendrite(D, window=W)
    x_hist = torch.randn(B, W, D)
    x_t = torch.randn(B, 1, D)
    Xp = torch.cat([x_hist, x_t], dim=1)

    D_par = dend.window_features(Xp, 1).squeeze(1)
    D_seq = dend.forward(x_hist, error=None)
    assert torch.allclose(D_par, D_seq, atol=1e-6)


def test_current_token_not_in_window():
    """Changing X_t must not change d_t (window is strictly past)."""
    torch.manual_seed(2)
    B, T, D, W = 1, 4, 8, 3
    dend = Dendrite(D, window=W)
    x_hist = torch.randn(B, W, D)
    X = torch.randn(B, T, D)
    Xp = torch.cat([x_hist, X], dim=1)
    d0 = dend.window_features(Xp, T)

    X2 = X.clone()
    X2[:, -1] += 10.0  # only last token changed
    Xp2 = torch.cat([x_hist, X2], dim=1)
    d1 = dend.window_features(Xp2, T)

    # positions 0..T-2 must be identical; only if window looked at current would last differ
    # actually last position's window is xp[T-1 : T-1+W] which includes X[T-1] only as
    # the *newest past* for the *next* step — for d_{T-1} the window ends at X[T-2].
    assert torch.allclose(d0[:, :-1], d1[:, :-1], atol=1e-6)
    # d at last position also must not see X[T-1] itself
    assert torch.allclose(d0[:, -1], d1[:, -1], atol=1e-6)


if __name__ == "__main__":
    test_window_features_matches_sequential()
    test_decode_T1_matches_forward()
    test_current_token_not_in_window()
    print("all dendrite parity tests passed")
