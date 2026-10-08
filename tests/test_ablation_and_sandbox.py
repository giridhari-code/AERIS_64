"""Production tests: ablation switches + sandbox isolation."""

from __future__ import annotations

import torch

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig
from neurofield.model import NeuroField
from neurofield.sandbox import SandboxConfig, SandboxExecutor, SandboxRegistry


def _cfg():
    return ModelConfig(
        d_model=32,
        d_k=16,
        d_v=16,
        n_skills=3,
        top_k=2,
        k_max=2,
        vocab_size=40,
        dendrite_window=2,
        tie_embeddings=True,
    )


def test_ablation_full_forward_and_grad():
    m = NeuroField(_cfg(), SafetyConfig(enable_audit=True), AblationConfig())
    m.train()
    x = torch.randint(0, 40, (2, 12))
    y = torch.randint(0, 40, (2, 12))
    out = m(x, targets=y)
    assert out.loss is not None
    out.loss.backward()
    assert any(p.grad is not None for p in m.fast_mem.parameters())


def test_ablation_no_fast_zeros_read_path():
    m = NeuroField(
        _cfg(),
        SafetyConfig(enable_audit=False),
        AblationConfig(use_fast_memory=False),
    )
    m.eval()
    with torch.no_grad():
        out = m(torch.randint(0, 40, (1, 8)), return_audit=True)
    assert out.logits.shape == (1, 8, 40)
    # write norms should be zeros when fast memory is off
    assert all(abs(v) < 1e-8 for v in (out.audit or {}).get("write_norms", [0.0]))


def test_ablation_set_hot_swap():
    m = NeuroField(_cfg())
    m.set_ablation(AblationConfig(use_slow_memory=False, use_neuromodulator=False, fixed_gate=0.0))
    out = m(torch.randint(0, 40, (1, 6)))
    assert out.logits.shape[0] == 1


def test_sandbox_allow_and_deny():
    ex = SandboxExecutor(SandboxConfig(timeout_sec=3), session_id="test")
    ok = ex.run_shell("echo ok")
    assert ok.ok and "ok" in ok.stdout
    denied = ex.run_shell("curl https://example.com")
    assert not denied.ok and "allow-list" in denied.stderr


def test_sandbox_registry_budget():
    reg = SandboxRegistry(max_calls_per_session=1)
    sb = reg.get("s1")
    assert sb.run_shell("echo a").ok
    second = sb.run_shell("echo b")
    assert not second.ok and "budget" in second.stderr
