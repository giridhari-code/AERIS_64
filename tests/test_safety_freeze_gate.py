"""Paper §7: anomaly z-score → memory_frozen → gate g=0; slow ring not written."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import SENTINEL
from neurofield.safety.monitor import SafetyMonitor


def _model():
    cfg = ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=4, top_k=2, k_max=2,
        vocab_size=40, dendrite_window=3, max_seq_len=32, dropout=0.0,
    )
    return SENTINEL(cfg, SafetyConfig(enable_audit=True, freeze_on_anomaly=True)).eval()


def test_memory_frozen_zeros_gate_in_audit():
    m = _model()
    x = torch.randint(0, 40, (1, 8))
    state = m.new_state(1, x.device)
    state["memory_frozen"] = True
    with torch.no_grad():
        out = m(x, state=state, return_audit=True)
    gates = (out.audit or {}).get("gates") or []
    assert gates, "expected gate audit"
    assert all(abs(float(g)) < 1e-6 for g in gates), gates
    assert out.audit.get("memory_frozen") is True


def test_frozen_skips_slow_ring_snapshots():
    m = _model()
    x = torch.randint(0, 40, (1, 16))
    state = m.new_state(1, x.device)
    state["memory_frozen"] = True
    with torch.no_grad():
        out = m(x, state=state, write_slow=True)
    assert float(out.state["slow_filled"].sum()) == 0.0


def test_monitor_freeze_after_alarms_per_session():
    mon = SafetyMonitor(z_threshold=0.0, freeze_after_alarms=2, min_history=30)
    # baseline for session A
    for _ in range(30):
        mon.update(
            {"gates": [0.5], "write_norms": [1.0], "mem_norms": [1.0], "router_entropy": [0.5], "surprises": [0.2]},
            session_id="A",
        )
    mon.update({"gates": [100.0], "write_norms": [1.0], "mem_norms": [1.0], "router_entropy": [0.5]}, session_id="A")
    mon.update({"gates": [100.0], "write_norms": [1.0], "mem_norms": [1.0], "router_entropy": [0.5]}, session_id="A")
    assert mon.should_freeze("A")
    assert not mon.should_freeze("B")  # other session not frozen


def test_monitor_reset_session_clears_freeze():
    mon = SafetyMonitor(z_threshold=0.0, freeze_after_alarms=1, min_history=5)
    for _ in range(5):
        mon.update({"gates": [0.5]}, session_id="S")
    mon.update({"gates": [99.0]}, session_id="S")
    assert mon.should_freeze("S")
    mon.reset_session("S")
    assert not mon.should_freeze("S")


def test_unfrozen_gate_not_all_zero():
    m = _model()
    x = torch.randint(0, 40, (1, 8))
    with torch.no_grad():
        out = m(x, return_audit=True)
    gates = (out.audit or {}).get("gates") or []
    assert gates
    assert any(abs(float(g)) > 1e-3 for g in gates)
