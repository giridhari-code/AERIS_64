"""Incremental decode must equal one full forward pass (works on the sequential AND the parallel model).

The server and eval suite decode like this: prefill the prompt, then feed one token at a time with `state`.
Training runs one full pass. If the two compute different functions, what is trained is not what is served.

Only the public API is used (no k_depth / write_slow arguments), so this runs on any version of NeuroField.
Depth is varied through ModelConfig.k_max. The slow ring is switched off by an ablation because it is written
at the end of every multi-token call and would change the slow read between a prefill and a full pass; that
effect is real but separate from the fast-memory / error-path question tested here.

By code reading (CODE_AUDIT_2026-10-12.md, finding H1) these tests are expected to FAIL on the parallel-main
model from the audited zip, and to PASS on a sequential per-token model. Not run by the author (no torch there).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig  # noqa: E402
from neurofield.model import NeuroField  # noqa: E402

V = 40


def _make(k_max: int):
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=k_max,
        vocab_size=V, max_seq_len=64, dendrite_window=3, dropout=0.0,
    )
    m = NeuroField(cfg, SafetyConfig(enable_audit=False, truncate_write_window=64))
    m.set_ablation(AblationConfig(use_slow_memory=False))
    return m.eval()


@pytest.mark.parametrize("k_max", [1, 2])
def test_prefill_then_one_token_step_matches_full_pass(k_max):
    torch.manual_seed(0)
    m = _make(k_max)
    x = torch.randint(0, V, (1, 12))
    with torch.no_grad():
        full = m(x).logits[:, -1]
        pre = m(x[:, :-1])
        step = m(x[:, -1:], state=pre.state).logits[:, -1]
    diff = (full - step).abs().max().item()
    assert diff < 1e-4, f"decode step differs from the full pass: max diff {diff:.4f} (k_max={k_max})"


def test_token_by_token_matches_full_pass():
    torch.manual_seed(1)
    m = _make(1)
    x = torch.randint(0, V, (1, 10))
    with torch.no_grad():
        full = m(x).logits
        state, outs = None, []
        for t in range(x.size(1)):
            o = m(x[:, t:t + 1], state=state)
            state = o.state
            outs.append(o.logits[:, -1])
        inc = torch.stack(outs, dim=1)
    diff = (full - inc).abs().max().item()
    assert diff < 1e-4, f"token-by-token decode differs from the full pass: max diff {diff:.4f}"
