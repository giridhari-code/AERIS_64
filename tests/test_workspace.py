"""Learned Global Workspace (J-space): neutrality, determinism, parity, sparsity, grads."""
from __future__ import annotations

import torch

from sentinel.config import AblationConfig, ModelConfig
from sentinel.model import SENTINEL
from sentinel.modules.workspace import GlobalWorkspace


def _cfg(**kw) -> ModelConfig:
    base = dict(
        d_model=64, d_k=32, d_v=32, n_skills=4, top_k=2, k_max=3,
        dendrite_window=3, vocab_size=128, max_seq_len=64,
        jspace_dim=32, jspace_slots=8,
        workspace_mode="learned", ws_atoms=32, ws_k_active=4, ws_rank=4,
    )
    base.update(kw)
    return ModelConfig(**base)


def _arm(m: SENTINEL) -> SENTINEL:
    """Make the workspace matter: W_js / ws_bias are zero-init by design."""
    with torch.no_grad():
        m.field.W_js.weight.normal_(0, 0.2)
        m.router.ws_bias.weight.normal_(0, 0.5)
        if m.workspace is not None:
            m.workspace.lens_out.weight.normal_(0, 0.05)
    return m


def _decode(m: SENTINEL, ids: torch.Tensor, **kw) -> torch.Tensor:
    st = m.new_state(ids.size(0), torch.device("cpu"))
    outs = []
    with torch.no_grad():
        for t in range(ids.size(1)):
            o = m(ids[:, t : t + 1], state=st, **kw)
            st = o.state
            outs.append(o.logits[:, 0])
    return torch.stack(outs, 1)


def test_rule_mode_keeps_old_state_dict_keys():
    m = SENTINEL(_cfg(workspace_mode="rule"))
    assert m.workspace is None
    assert not any(k.startswith("workspace.") for k in m.state_dict())


def test_learned_workspace_is_exactly_neutral_at_init():
    torch.manual_seed(0)
    rule = SENTINEL(_cfg(workspace_mode="rule")).eval()
    torch.manual_seed(0)
    learned = SENTINEL(_cfg()).eval()
    missing, unexpected = learned.load_state_dict(rule.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("workspace.") for k in missing)
    ids = torch.randint(0, 128, (2, 14))
    with torch.no_grad():
        a = rule(ids, k_depth=2).logits
        b = learned(ids, k_depth=2).logits
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_learned_workspace_is_deterministic_and_session_isolated():
    torch.manual_seed(1)
    m = _arm(SENTINEL(_cfg())).eval()
    ids = torch.randint(0, 128, (1, 12))
    a = _decode(m, ids)
    a2 = _decode(m, ids)
    _decode(m, torch.randint(0, 128, (1, 40)))  # another "user" on the same model object
    b = _decode(m, ids)
    assert (a - a2).abs().max().item() == 0.0
    assert (a - b).abs().max().item() == 0.0


def test_learned_workspace_stays_deterministic_with_adaptive_decode_depth():
    # opt-in per-token depth (branch fix/sentinel-debug): the metacognition-driven k must not leak either
    torch.manual_seed(11)
    m = _arm(SENTINEL(_cfg(adaptive_decode_depth=True))).eval()
    ids = torch.randint(0, 128, (1, 12))
    a = _decode(m, ids)
    _decode(m, torch.randint(0, 128, (1, 40)))
    b = _decode(m, ids)
    assert (a - b).abs().max().item() == 0.0


def test_full_sequence_equals_token_by_token_with_workspace_on():
    torch.manual_seed(2)
    m = _arm(SENTINEL(_cfg())).eval()
    ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        full = m(ids, k_depth=2).logits
    step = _decode(m, ids, k_depth=2)
    assert (full - step).abs().max().item() < 1e-4


def test_code_is_sparse_nonnegative_and_capacity_limited():
    torch.manual_seed(3)
    ws = GlobalWorkspace(d_model=64, d_ws=32, n_atoms=32, k_active=4, rank=4)
    h = torch.randn(3, 9, 64)
    b, c = ws(h)
    assert b.shape == (3, 9, 32) and c.shape == (3, 9, 32)
    assert (c >= 0).all()
    assert int((c > 0).sum(-1).max()) <= 4


def test_workspace_params_get_gradients_once_it_is_wired_in():
    torch.manual_seed(4)
    m = _arm(SENTINEL(_cfg())).train()
    x = torch.randint(0, 128, (2, 16))
    out = m(x, targets=x)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for name in ("atoms",):
        g = getattr(m.workspace, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    for p in (m.workspace.broadcast.weight, m.workspace.lens_out.weight, m.workspace.lens_in.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert "ws_active" in out.metrics and 0 < out.metrics["ws_active"] <= 4


def test_lens_loss_trains_the_lens_even_with_zero_init_consumers():
    torch.manual_seed(5)
    m = SENTINEL(_cfg(lambda_lens=0.1)).train()  # W_js / ws_bias still zero
    x = torch.randint(0, 128, (2, 24))
    out = m(x, targets=x)
    assert out.metrics["lens_loss"] > 0
    out.loss.backward()
    g = m.workspace.lens_out.weight.grad
    assert g is not None and g.abs().sum() > 0  # gradient reached J only through the lens loss


def test_ablation_use_workspace_false_deletes_the_broadcast():
    torch.manual_seed(6)
    m = _arm(SENTINEL(_cfg())).eval()
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        on = m(ids, k_depth=2).logits
        m.set_ablation(AblationConfig(use_workspace=False))
        off = m(ids, k_depth=2).logits
        m.field.W_js.weight.zero_()
        m.router.ws_bias.weight.zero_()
        zero = m(ids, k_depth=2).logits
    assert (on - off).abs().max().item() > 1e-4          # workspace really mattered
    assert torch.allclose(off, zero, atol=1e-6)           # ablated == no consumers


def test_steering_the_workspace_changes_the_output():
    torch.manual_seed(7)
    m = _arm(SENTINEL(_cfg())).eval()
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        base = m(ids, k_depth=2).logits
        steer = torch.zeros(m.workspace.n_atoms)
        steer[0] = 5.0
        m._ws_steer = steer
        steered = m(ids, k_depth=2).logits
        m._ws_steer = None
    assert (base - steered).abs().max().item() > 1e-4


def test_reportability_api_shapes():
    torch.manual_seed(8)
    m = SENTINEL(_cfg()).eval()
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        m(ids, k_depth=2)
    code = m._last_ws_code
    assert code is not None and code.shape == (8, 32)
    h = torch.randn(5, 64)
    tok, p = m.workspace.read_tokens(h, m.head.weight, topn=6)
    assert tok.shape == (5, 6) and p.shape == (5, 6)
    assert (p.sum(-1) <= 1.0 + 1e-5).all()
    words = m.workspace.atom_words(m.head.weight, topn=3)
    assert words.shape == (32, 3)
