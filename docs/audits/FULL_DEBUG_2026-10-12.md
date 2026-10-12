# Full component + training debug — 2026-10-12

## Verified green

| Area | Result |
|------|--------|
| Parallel router/field/read_seq | On main; `parallel_ops` wired |
| Parallel vs serial loss | Exact after `bal_loss` mean(0) fix |
| Parallel vs serial logits | ~1e-5 float order noise (atol 2e-5 in test) |
| Decode parity / leakage | Pass |
| Train smoke (tiny, 20 steps) | `gate~0.88`, checkpoint saves |
| Checkpoint reload + infer | OK |
| `collect_stats` / gate_mean metrics | OK |

## Fixes in this pass

1. **bal_loss** in parallel path: `pi.mean(dim=(0,1))` → `pi.mean(0)` to match per-token loop (affects loss/grads).
2. Parallel identity test: logits atol `2e-5` (float reduction order).

## Not in scope (by design)

- Delta-rule chunk-parallel scan (would change numerics; sequential remains).
- DDP multi-rank test may be slow on some hosts; logic unchanged.
