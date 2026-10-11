# Paper v3 ↔ code alignment

Source: *AESC v2: Two-Speed Memory Architecture (Revised Technical Report, v3)* (2026-10-11).

## Failure modes (Section 9) — code status

| # | Failure | Code status |
|---|---------|-------------|
| 1 | Fast weights averaged over batch | **Fixed** — `state["M"]` is `(B, d_k, d_v)` |
| 2 | Writes frozen at inference | **Fixed** — writes run in `eval()`; session-scoped state |
| 3 | Neuromodulator off the graph | **Fixed** — `g` multiplies delta-rule write; grads reach gate |
| 4 | Dendrite as autoencoder | **Fixed** — predictor forecasts `x_{t+1}`; error one step late |
| 5 | Router weights ignored | **Fixed** — STE: `w + π - sg(π)` in `SkillRouter` |
| 6 | Field without memory read | **Fixed** — `field(h, d_t, r, k)` every token |
| 7 | Forgetting controller unused | **Fixed** — learnable `log_lambda` per key in `FastMemory` |
| 8 | Vocabulary swallows budget | **Mitigated** — `tie_embeddings` default True |
| 9 | Stop-gradient write blocks learning | **Fixed** — differentiable write inside truncate window |
| 10 | Label leakage | **Fixed + tested** — delayed error; `tests/test_paper_checklist.py` |
| 11 | Figure errors | Docs only |

## Appendix B checklist

Covered by `tests/test_paper_checklist.py`, `tests/test_model.py`, `tests/test_slow_context_ring.py`.

## Explicitly unproven (paper Section 10)

Slow memory, neuromodulator, metacognition, router, predictor **contribution** on tasks other than 4-pair recall — no claim in code either. Ablation harness exists (`AblationConfig`, `scripts/research/contribution_ablation.py`).

## Future work (not implemented)

Per-user `C_u` (Section 11) — proposal only; session `M` + slow ring partially cover short context, not durable cross-session private memory.

## Training entry (v3 fix)

`scripts/train/train_company.py` now runs paper slow-memory replay every 50 steps:

```
W_s ← (1-η) W_s + η Σ_b (S^b / Σ S) M^b
```

Previously only `Trainer` in `training/trainer.py` did this; the company script omitted it.

