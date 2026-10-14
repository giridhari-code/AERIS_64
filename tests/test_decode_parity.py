"""Train == generate: one whole-sequence call, chunked calls and token-by-token decode must be the SAME function.

Regression for the 'H1' bug: decode used E = X (the previous prediction was not carried, so the shifted X_hat was zero)
and, for k_depth > 1, depth passes >= 1 restarted their fast memory from zero at every decode step.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig  # noqa: E402
from neurofield.model import SENTINEL  # noqa: E402

V = 40


def _model(seed=0, slow=False, **kw):
    cfg = ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=4, top_k=2, k_max=3, vocab_size=V,
                      dendrite_window=3, max_seq_len=64, dropout=0.0)
    torch.manual_seed(seed)
    return SENTINEL(cfg, SafetyConfig(enable_audit=False), AblationConfig(use_slow_memory=slow, **kw)).eval()


def _decode(m, x, split, k_depth, write_slow=True):
    """prefill x[:, :split] in one call, then one token per call with the state (what the server does)."""
    with torch.no_grad():
        out = m(x[:, :split], k_depth=k_depth, write_slow=write_slow) if split > 0 else None
        st, logits = (out.state, [out.logits]) if out else (None, [])
        for t in range(split, x.size(1)):
            o = m(x[:, t : t + 1], state=st, k_depth=k_depth, write_slow=write_slow)
            st = o.state
            logits.append(o.logits)
    return torch.cat(logits, dim=1), st


@pytest.mark.parametrize("k_depth", [1, 2, 3])
@pytest.mark.parametrize("split", [0, 1, 5, 11])
def test_token_by_token_decode_equals_the_whole_sequence_pass(k_depth, split):
    m = _model()
    x = torch.randint(0, V, (2, 14))
    with torch.no_grad():
        whole = m(x, k_depth=k_depth).logits
    dec, _ = _decode(m, x, split, k_depth)
    assert torch.allclose(whole, dec, atol=1e-5, rtol=1e-5), (whole - dec).abs().max()


@pytest.mark.parametrize("k_depth", [1, 3])
def test_parity_holds_with_slow_memory_enabled_when_the_ring_is_not_written(k_depth):
    m = _model(slow=True)
    x = torch.randint(0, V, (1, 13))
    with torch.no_grad():
        whole = m(x, k_depth=k_depth, write_slow=False).logits
    dec, _ = _decode(m, x, 4, k_depth, write_slow=False)
    assert torch.allclose(whole, dec, atol=1e-5)


def test_error_term_is_x_minus_previous_prediction_not_the_whole_embedding():
    """At a decode step E must be X_t - f_p(H_{t-1}); with the old bug it was X_t, so state['error'] == embed(x_t)."""
    m = _model()
    x = torch.randint(0, V, (1, 6))
    with torch.no_grad():
        st = m(x[:, :5], k_depth=1).state
        o = m(x[:, 5:6], state=st, k_depth=1)
        X_t = m.embed(x[:, 5])
        assert not torch.allclose(o.state["error"], X_t, atol=1e-4)           # not the whole embedding any more
        expected = X_t - m.predictor(st["Hin"][0])                            # X_t - X_hat made at t-1
        assert torch.allclose(o.state["error"], expected, atol=1e-5)


def test_fast_memory_accumulates_in_every_depth_pass_across_decode_steps():
    m = _model()
    x = torch.randint(0, V, (1, 12))
    with torch.no_grad():
        whole_state = m(x, k_depth=3).state
    _, dec_state = _decode(m, x, 0, 3)
    for j in range(3):
        assert float(dec_state["M_pass"][j].norm()) > 0
        assert torch.allclose(dec_state["M_pass"][j], whole_state["M_pass"][j], atol=1e-5), f"pass {j} memory differs"
        assert torch.allclose(dec_state["R_pass"][j], whole_state["R_pass"][j], atol=1e-5)
    # the passes really are different memories (passes >= 1 are not just a copy of pass 0)
    assert not torch.allclose(dec_state["M_pass"][0], dec_state["M_pass"][2], atol=1e-4)


def test_greedy_generation_equals_teacher_forced_training_forward():
    """The property that matters: what the server generates is what the trained forward would predict."""
    m = _model()
    prompt = torch.randint(0, V, (1, 7))
    with torch.no_grad():
        out = m(prompt)
        st, gen = out.state, []
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(8):
            gen.append(tok)
            o = m(tok, state=st)
            st, tok = o.state, o.logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat(gen, dim=1)
        full = torch.cat([prompt, gen], dim=1)
        tf = m(full[:, :-1]).logits[:, prompt.size(1) - 1 :].argmax(-1)       # one whole-sequence pass
    assert torch.equal(tf, gen)


def test_state_from_another_k_depth_is_ignored_not_misused():
    """A carried per-pass state is only reused for the SAME k_depth; otherwise the passes restart from zero
    (the shared dendrite history x_hist is still carried, so compare against a state stripped of the per-pass keys)."""
    m = _model()
    x = torch.randint(0, V, (1, 9))
    with torch.no_grad():
        st2 = m(x[:, :5], k_depth=2).state
        o = m(x[:, 5:], state=st2, k_depth=3)           # different depth: no crash, passes restart from zero
        stripped = dict(st2, k_depth=None, M_pass=None, Hin=None, D_pass=None, R_pass=None)
        ref = m(x[:, 5:], state=stripped, k_depth=3)
    assert o.state["k_depth"] == 3 and len(o.state["M_pass"]) == 3
    assert torch.allclose(o.logits, ref.logits, atol=1e-6)


def test_training_forward_on_a_fresh_sequence_still_backpropagates_through_every_pass():
    m = _model().train()
    x, y = torch.randint(0, V, (2, 10)), torch.randint(0, V, (2, 10))
    out = m(x, targets=y, k_depth=3)
    out.loss.backward()
    assert float(m.fast_mem.W_k.weight.grad.abs().sum()) > 0 and float(m.predictor.net[0].weight.grad.abs().sum()) > 0
