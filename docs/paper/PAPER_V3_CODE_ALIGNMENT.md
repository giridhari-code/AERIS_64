# AESC paper v3.1 ↔ code alignment

Report: *AESC v2: Two-Speed Memory Architecture (Revised Technical Report, v3.1)*  
Repo HEAD at alignment: see `git log -1`.

## Eleven failure modes → code status

| # | Failure | Fix in paper | Code status |
|---|---------|--------------|-------------|
| 1 | Fast M averaged over batch | Per-sequence `(B,d_k,d_v)` | **OK** — `state["M"]` zeros per batch |
| 2 | Writes frozen at inference | Gated writes at infer | **OK** — write not training-only |
| 3 | Neuromodulator off graph | Gate × differentiable write | **OK** — `g` multiplies delta write |
| 4 | Dendrite as autoencoder | Next-input predict; error 1-step late | **OK** — `pending_pred` + detach target |
| 5 | Router weights ignored | Weighted skills + STE | **OK** — `router.py` straight-through |
| 6 | Field without memory read | `r` into field | **OK** — field gets memory read |
| 7 | Forgetting controller unused | Learnable per-key decay | **OK** — `log_lambda` → Λ |
| 8 | Vocab swallows budget | Tied embeddings | **OK** — `tie_embeddings=True` default |
| 9 | Stop-grad write blocks learning | Differentiable write + truncate window | **OK** — truncate BPTT window |
| 10 | Label leakage | One-step delay | **OK** by design; unit test still recommended |
| 11 | Figure 1 errors | Docs redrawn | Docs / paper |

## Paper claims honesty (from report)

- **C1** fast memory recall: supported in paper’s 9.92M run  
- **C2** other blocks help: **untested** in paper  
- Parallel / chunk forms: implemented in recent commits; equivalence tests ongoing  

## Recent code (already on main)

- Paper v3 PDF under `docs/paper/`
- Token-parallel / chunk train, DDP path, GPU optim
- Slow ring compaction tests
- Field R / replay on train entrypoints

## Still open (paper + engineering)

1. Leakage **unit test** (Appendix B) automated in CI  
2. C2 ablations on tasks built for each block  
3. ~~Full safety observability~~ **DONE**: anomaly → state memory_frozen → g=0 (§7)  
4. Multi-seed / baseline comparisons  

*This file tracks alignment; it does not claim the architecture is proven end-to-end.*
