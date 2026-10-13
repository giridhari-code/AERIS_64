#!/usr/bin/env python3
"""Print paper alignment smoke status for AERIS_64 reference (looped) form."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from neurofield.config import ModelConfig, SafetyConfig
from neurofield.model import NeuroField
from neurofield.modules.memory import FastMemory


def main() -> int:
    ok = 0
    fail = 0

    def check(name: str, cond: bool) -> None:
        nonlocal ok, fail
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}")
        if cond:
            ok += 1
        else:
            fail += 1

    print("AESC v3.1 ↔ code alignment smoke (reference looped form)\n")
    m = NeuroField(
        ModelConfig(
            d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=2,
            vocab_size=40, dendrite_window=2, max_seq_len=32, dropout=0.0,
        ),
        SafetyConfig(enable_audit=False),
    )
    check("parallel_ops forced True", m.parallel_ops is True)
    x = torch.randint(0, 40, (2, 8))
    out = m(x, targets=x)
    check("forward path == parallel", out.metrics.get("path") == "parallel")
    check("loss finite", out.loss is not None and torch.isfinite(out.loss))

    st = m.new_state(2, torch.device("cpu"))
    st["memory_frozen"] = True
    out_f = m(x, state=st, collect_stats=True)
    check("frozen gate_mean ~ 0", float(out_f.metrics.get("gate_mean", 1.0)) < 1e-5)

    fm = FastMemory(16, 8, 8)
    M = torch.randn(1, 8, 8)
    d = torch.randn(1, 16)
    o = torch.randn(1, 16)
    g0 = torch.zeros(1, 1)
    M2 = fm.write(M, d, o, g0)
    lam = fm.lambda_diag.view(1, -1, 1)
    check("g=0 only decays", torch.allclose(M2, lam * M, atol=1e-5))

    print(f"\nResult: {ok} passed, {fail} failed")
    print("Deferred (not failures): chunkwise Phase1/Phase2 train, full 800-step 9.92M recall claim.")
    print("Checklist: docs/audits/PAPER_CODE_ALIGNMENT_100.md")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
