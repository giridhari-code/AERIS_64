# Memory fixes (scale / isolation)

## Fixed

| Issue | Fix |
|-------|-----|
| Write-norm **batch-global** | Per-sequence `norm(dim=(-2,-1))` + scale `(B,1,1)` |
| Decay init too fast (`sigmoid(2)≈0.88`) | `log_lambda = linspace(1, 6)` multi-timescale |
| Error target pulls embeddings | `target.detach()` in predictor error |

## Verify

```bash
PYTHONPATH=src python scripts/ablate_memory.py
```

## Still design risks (not all fixed)

- Slow memory basis vs fast `W_k`
- Single `d_k×d_v` capacity at 1B
- `choose_k` still uses batch mean (OK for B=1 serve)
- BPTT: `M` detach window vs full-`h` gradient

See analysis notes; Run-1 results not invalidated.
