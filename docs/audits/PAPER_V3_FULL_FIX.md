# Paper v3 full code fix (2026-10-11)

## Real gaps closed

1. **NeuralField R** — was `Conv1d` on length-1 (broken local recurrence). Now convolves over **neuron index** (D), matching Appendix A “local convolution over neighbouring neurons”.
2. **Slow replay** — `train_company` + `train_skills` + `train_multilang` + `train_align` call `maybe_replay` every 50 steps (Appendix A W_s update). Shared helper: `neurofield.training.paper_step`.
3. **Tests** — `tests/test_associative_recall_smoke.py` (field R shape + loss drops on tiny recall).

## Already correct (verified)

| Appendix A / B item | Code |
|---------------------|------|
| Per-seq M | OK |
| Delayed e_t | OK |
| STE router | OK |
| Fast write key=d_{t-1} | OK |
| Field + W_m r | OK |
| Gate formula + δ | OK |
| Inference writes | OK |
| Leakage tests | OK |

## Still not in scope (paper Section 11)

- Per-user C_u
- Equal-FLOP baselines vs Transformer
