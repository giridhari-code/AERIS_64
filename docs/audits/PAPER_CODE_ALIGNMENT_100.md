# Paper ↔ code alignment (target ~100% of *claimable* surface)

Paper: AESC v2 Technical Report v3.1 · Code: AERIS_64 main (looped form)

## A. Core equations — DONE

| Claim | Code | Test |
|-------|------|------|
| Dendrite window + gamma * e | `modules/dendrite.py`, `model.py` | components / model |
| Predictor delayed error (shift) | `model.py` parallel depth | decode parity / leakage |
| Gate g + freeze ⇒ g=0 | neuromod + `memory_frozen` | `test_safety_freeze_gate` |
| Change-5 decay-consistent delta | `FastMemory.write` | `test_delta_rule_change5` |
| g=0 ⇒ only Lambda * M | same | `test_zero_gate_only_decays` |
| Router top-k + STE | `modules/router.py` | router unit / model |
| Field single_step | `modules/field.py` | model |
| Slow ring read/write | `SlowMemory` | slow context tests |
| Looped depth over T | `model.forward` only path | `test_parallel_main` |
| Token-serial outer path removed | `parallel_ops=True` forced | model docstring |

## B. Training hyperparams (paper-aligned defaults)

| Paper | Code |
|-------|------|
| Replay every ~50–100 steps | `replay_every = 50` in `train_company.py` |
| ~2% step toward surprise-weighted M | `replay_eta = 0.02` |
| CE + aux losses | lambda_err / lambda_bal in forward |
| Label smoothing / dropout optional | CLI flags |

## C. Explicitly NOT claimed as done (paper also says untested)

| Item | Status |
|------|--------|
| Chunkwise Phase1 sequential / Phase2 parallel train | **Deferred** — not required for alignment of *reference form* |
| Chunkwise + diagonal Lambda + per-token g exact | **Untested** (paper §) |
| 10M all-modules non-zero grad on full run | Re-run on your GPU when training |
| Associative recall 1.000 @ 800 steps @ 9.92M | Smoke test only in CI; full claim is experiment log |

## D. Failure-mode checklist (Section 9 style)

| # | Mode | Mitigation in code |
|---|------|-------------------|
| 1 | Fast M shared across batch | M is per-batch-row `(B,dk,dv)` |
| 2 | Decode ≠ train path | Single `forward`; T=1 + state |
| 3 | Future leakage | shift error, causal write_seq, tests |
| 4 | Delta vs decay inconsistent | Change-5 write |
| 5 | Router grads / equal average | STE path in router |
| 6 | Gate stuck | monitor gate_mean; freeze only on flag |
| 7 | Slow pollution from live infer | replay from train path only |
| 8 | Empty data train | refuse if tokens < 8 |
| 9 | Parallel depth M leak | fresh M per depth pass |
| 10 | End flush breaks parity | no T>1 end flush |
| 11 | Zero-grad modules | ablation flags + train real data |

## E. How to verify on your machine

```bash
git pull origin main
PYTHONPATH=src python -m pytest tests/test_delta_rule_change5.py tests/test_safety_freeze_gate.py tests/test_parallel_main.py tests/test_decode_parity.py tests/test_paper_alignment_v31.py -q
PYTHONPATH=src python scripts/eval/paper_alignment_smoke.py
```

Alignment of **reference looped form** is treated complete; chunkwise speed path remains future work.
