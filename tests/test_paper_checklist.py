"""Appendix B items of the v2 report that had no test: leakage / causality and session isolation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurofield.config import ModelConfig, SafetyConfig  # noqa: E402
from neurofield.model import NeuroField  # noqa: E402

V = 50


def _model(**kw):
    cfg = ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=4, top_k=2, k_max=3,
                      vocab_size=V, dendrite_window=3, max_seq_len=64, dropout=0.0, **kw)
    return NeuroField(cfg, SafetyConfig(enable_audit=False))


@pytest.mark.parametrize("batch", [1, 3])
def test_no_label_leakage_future_tokens_do_not_change_past_logits(batch):
    """Checklist 8 / Proposition 7: changing x(t+1..) must not change logits at positions <= t."""
    torch.manual_seed(0)
    m = _model().eval()  # default settings: no manual track_global switch
    x = torch.randint(0, V, (batch, 20))
    with torch.no_grad():
        base = m(x).logits
        for t in (0, 3, 7, 12, 18):
            x2 = x.clone()
            x2[:, t + 1 :] = torch.randint(0, V, x2[:, t + 1 :].shape)
            alt = m(x2).logits
            assert torch.equal(base[:, : t + 1], alt[:, : t + 1]), f"future token leaked into logits at t<={t}"


def test_targets_do_not_change_logits():
    torch.manual_seed(1)
    m = _model().eval()
    x = torch.randint(0, V, (2, 16))
    with torch.no_grad():
        a = m(x).logits
        b = m(x, targets=torch.randint(0, V, (2, 16))).logits
    assert torch.equal(a, b)


def test_eval_is_deterministic_and_isolated_between_calls_by_default():
    """Checklist 1 / Section 6 control 1: with default settings the 2nd call must not inherit the 1st."""
    torch.manual_seed(2)
    m = _model().eval()
    assert m.meta.track_global is True  # default is unchanged; eval() must still be isolated
    x = torch.randint(0, V, (1, 20))
    with torch.no_grad():
        a = m(x).logits
        m(torch.randint(0, V, (1, 20)))  # "another user"
        b = m(x).logits
    assert torch.equal(a, b)
