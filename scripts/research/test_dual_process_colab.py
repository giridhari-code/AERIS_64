#!/usr/bin/env python3
"""Colab / local smoke: Metacognition dual-process (System-1 / System-2).

Run:
  PYTHONPATH=packages python scripts/research/test_dual_process_colab.py
"""
from __future__ import annotations

import time
from typing import Any

import torch

from sentinel.modules.metacognition import Metacognition


def _signals(surprise: float, conflict: float, batch: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic monitor signals (no full model required)."""
    S = torch.full((batch,), float(surprise))
    # step() expects prediction-error style inputs — we call dual_process directly + full step
    conf = torch.sigmoid(-2.0 * (S - 1.0))
    confl = torch.full((batch,), float(conflict))
    return S, confl, conf


def trial(name: str, surprise: float, conflict: float, force: str | None = None) -> dict[str, Any]:
    meta = Metacognition(k_max=5, track_global=False)
    meta.training = False
    rel_S = torch.full((4,), surprise)
    confl = torch.full((4,), conflict)
    conf = torch.sigmoid(-2.0 * (rel_S - 1.0))
    t0 = time.perf_counter()
    mode, k = meta.dual_process(rel_S, confl, conf, force=force)
    # also run a fake step path via allocate only for parity
    dt = (time.perf_counter() - t0) * 1000
    return {
        "case": name,
        "surprise": surprise,
        "conflict": conflict,
        "force": force or "auto",
        "process_mode": mode,
        "k": k,
        "ms": round(dt, 4),
    }


def main() -> None:
    cases = [
        trial("easy_greeting", surprise=0.2, conflict=0.1),
        trial("simple_fact", surprise=0.4, conflict=0.2),
        trial("ambiguous", surprise=1.2, conflict=0.5),
        trial("hard_conflict", surprise=1.8, conflict=1.2),
        trial("force_s1_on_hard", surprise=1.8, conflict=1.2, force="system1"),
        trial("force_s2_on_easy", surprise=0.2, conflict=0.1, force="system2"),
    ]
    print("=== Metacognition dual-process (System-1 / System-2) ===\n")
    print(f"{'case':<22} {'mode':<10} {'k':>3}  surprise  conflict  force")
    print("-" * 72)
    for r in cases:
        print(
            f"{r['case']:<22} {r['process_mode']:<10} {r['k']:>3}  "
            f"{r['surprise']:>8.2f}  {r['conflict']:>8.2f}  {r['force']}"
        )
    s1 = sum(1 for r in cases if r["process_mode"] == "system1")
    s2 = sum(1 for r in cases if r["process_mode"] == "system2")
    print(f"\nAuto+force counts: system1={s1} system2={s2}")
    # Assert expected
    assert cases[0]["process_mode"] == "system1" and cases[0]["k"] == 1
    assert cases[3]["process_mode"] == "system2" and cases[3]["k"] >= 2
    assert cases[4]["process_mode"] == "system1" and cases[4]["k"] == 1
    assert cases[5]["process_mode"] == "system2" and cases[5]["k"] == 5
    print("\nOK — dual-process thresholds behave as expected.")


if __name__ == "__main__":
    main()
