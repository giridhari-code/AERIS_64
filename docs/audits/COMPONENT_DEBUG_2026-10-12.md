# One-by-one component debug — 2026-10-12

HEAD at audit: `e97d1f9` (+ dendrite length assert).

| # | Component | Status | Notes |
|---|-----------|--------|-------|
| 1 | CharTokenizer | PASS | No orphan specials; specials only if data has them |
| 2 | Dendrite | PASS | `window_features` == per-token loop (maxdiff 0); needs xp len = W+T |
| 3 | NeuralField | PASS | R shape (1,1,3); grads OK |
| 4 | Predictor | PASS | Surprise detached |
| 5 | FastMemory | PASS | Key/val/gate grads |
| 6 | SlowMemory | PASS | `read_seq` == loop; compact ring grows filled weights by design |
| 7 | Neuromodulator | PASS | Gate not stuck at 0 |
| 8 | Metacognition | PASS | k + gate_bias |
| 9 | SkillRouter | PASS | STE + π sum 1 |
| 10 | Full Parallel Form | PASS | gate~0.88, no leak, decode parity, parallel≈serial |

False alarms cleared: tokenizer roundtrip only for in-vocab chars; ring “wrap” with compaction does not keep filled_sum == n_slots.
