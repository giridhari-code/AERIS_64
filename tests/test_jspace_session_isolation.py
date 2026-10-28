"""Rule-mode J-space must not leak across sessions (engine design fix)."""

from __future__ import annotations

import torch

from sentinel.config import ModelConfig
from sentinel.model import SENTINEL


def test_rule_jspace_no_cross_session_logit_leak():
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=2,
        dendrite_window=2, vocab_size=64, max_seq_len=64,
        jspace_dim=16, jspace_slots=8, workspace_mode="rule",
    )
    m = SENTINEL(cfg).eval()
    m.meta.track_global = False

    def run(prompt, steps=4):
        out = m(torch.tensor([prompt], dtype=torch.long), state=None, collect_stats=False)
        state = out.state
        logits0 = out.logits[0, -1].detach().clone()
        for _ in range(steps - 1):
            tok = int(out.logits[0, -1].argmax().item())
            out = m(torch.tensor([[tok]], dtype=torch.long), state=state, collect_stats=False)
            state = out.state
        return logits0, state

    a0, st_a = run([1, 2, 3, 4, 5, 6])
    run([20, 21, 22, 23])
    a0b, _ = run([1, 2, 3, 4, 5, 6])
    assert (a0 - a0b).abs().max().item() < 1e-5
    assert m.meta._js_slots is None
    assert st_a.get("jspace") is not None
