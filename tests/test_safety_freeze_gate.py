"""Paper §7: memory_frozen forces gate g=0 (writes only decay)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField
from neurofield.safety.monitor import SafetyMonitor


def test_memory_frozen_zeros_gate_in_audit():
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=4, top_k=2, k_max=2,
        vocab_size=40, dendrite_window=3, max_seq_len=32, dropout=0.0,
    )
    m = NeuroField(cfg, SafetyConfig(enable_audit=True)).eval()
    x = torch.randint(0, 40, (1, 8))
    state = m.new_state(1, x.device)
    state["memory_frozen"] = True
    with torch.no_grad():
        out = m(x, state=state, return_audit=True)
    gates = (out.audit or {}).get("gates") or []
    assert gates, "expected gate audit"
    assert all(abs(float(g)) < 1e-6 for g in gates), gates


def test_monitor_freeze_after_alarms():
    mon = SafetyMonitor(z_threshold=0.0, freeze_after_alarms=2)
    # feed enough history then spike
    for _ in range(30):
        mon.update({"gates": [0.5], "write_norms": [1.0], "mem_norms": [1.0], "router_entropy": [0.5]})
    mon.update({"gates": [100.0], "write_norms": [1.0], "mem_norms": [1.0], "router_entropy": [0.5]})
    mon.update({"gates": [100.0], "write_norms": [1.0], "mem_norms": [1.0], "router_entropy": [0.5]})
    assert mon.should_freeze()
