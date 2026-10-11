# Parallel Form is the main path

Sequential per-token time loop has been **removed**.

```text
X = Embed(tokens)
D_win = DendriteWindow(X)
H = D_win
for j in 1..k:          # depth
    E = sg(X) - shift(Predict(H))
    D = D_win + γ E
    O = Router(H)
    M,R = delta-rule over T (fresh M each depth)
    H = FieldStep(H, D, shift(R))
logits = Readout(O, R, U)
```

**Distributed (DDP)** is unchanged: multi-GPU training still uses this same forward on each rank.

Leakage: changing `x[t+1]` must not change logits at positions `≤ t` (tested).
