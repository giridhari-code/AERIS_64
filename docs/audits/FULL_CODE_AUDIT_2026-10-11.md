# Full code audit — 2026-10-11

## Scope

Runtime + static check of `main` against AESC paper v3 and production serve path.

## Automated

| Check | Result |
|-------|--------|
| `pytest tests/` | **64 passed** |
| Paper checklist (leakage, grads, inference writes, replay) | pass |
| Slow-ring unit tests | pass |
| SDK client (mock transport) | pass |
| Server / sandbox tests | pass |

## Architecture (manual + runtime)

| Item | Status |
|------|--------|
| Per-sequence fast M `(B,d_k,d_v)` | OK |
| Inference gated writes | OK |
| Gate × write + nonzero grads | OK |
| Delayed prediction error | OK |
| STE router | OK |
| Field reads `r` each step | OK |
| Learnable λ decay | OK |
| No label leakage (future x) | OK |
| Slow ring + paper replay in `train_company` | OK |
| Session state for decode | OK |
| Old checkpoint missing `log_temp` | soft-compat OK |
| UI `prompt` / `max_new_tokens` | OK |
| Package version 2.3.1 matches pyproject | OK |

## Fixes applied this audit

1. Metrics dict: use `.detach().item()` so logging does not touch the live autograd graph (UserWarning under train).

## Residual risks (not bugs)

- Slow / neuromod / metacog **contribution** still unproven on hard tasks (paper Section 10).
- Per-user `C_u` not implemented (paper Section 11 future work).
- Char-level small models need substantial data for open-ended chat quality.
