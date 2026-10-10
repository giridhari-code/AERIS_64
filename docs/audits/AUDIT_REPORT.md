> **Superseded by [AUDIT_V2_2.md](AUDIT_V2_2.md).** This file is the project's own earlier self-report; several of its claims (e.g. session isolation, tests) did not hold up under review.

# NeuroField v2 — Post-Audit Fix Report

**Date:** 2026-10-06

## Paper-faithful fixes applied

| # | Issue | Fix |
|---|--------|-----|
| 1 | Fast-memory value `v = W_v(o_t + d_t)` | **Paper:** `v = W_v(o_t)` only |
| 2 | Slow replay used same CE for all samples | **Per-sequence surprise** `_seq_surprise` passed to `replay_slow` |
| 3 | `bar_S` shared across users | **Per-session** `bar_S` in state; meta reset on new session |
| 4 | `max_write_norm` unused; freeze not wired | Write delta **clamped**; server **monitor → freeze_writes** |
| 5 | `param_report` double-counted tied weights | **Unique parameter** count by `id(p)` |

## Status after fixes

| Component | Status |
|-----------|--------|
| Core architecture | Matches Appendix A |
| Causality / leakage | Good |
| Fast-memory value equation | **Fixed** |
| Slow replay weighting | **Fixed** |
| Session isolation (`bar_S`) | **Fixed** |
| Write-norm cap | **Fixed** |
| Anomaly freeze path | **Wired** (enabled when `freeze_on_anomaly=true`) |
| Parameter reporting | **Fixed** |
| Unit tests | 3/3 pass |

## Claims (conservative)

- Fast memory **materially improves** associative recall on the synthetic task used in evidence scripts.
- Safety layer is implemented and wired; treat as **design + basic enforcement**, not fully production-hardened.

## How to serve with model loaded

```bash
CHECKPOINT=docs/neurofield_trained.pt PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/ for the chat frontend.
