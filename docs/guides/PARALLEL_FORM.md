# Parallel Form

Token-parallel **depth** path for AESC / NeuroField (paper §3).

## Evidence (honest)

- Sequential form: 4-pair associative recall **1.000** (existing reference).
- Parallel Form: same toy task reached **1.000** in a single-seed run and **passed leakage** (changing \(x_{t+1}\) does not change logits at \(t\)).
- Still early: multi-seed, harder tasks, and full baselines are open. Paper warning that “reference results do not carry over until tested” is addressed for this one task only.

## API

```python
# sequential (default, unchanged)
out = model(input_ids, targets=y)

# parallel form
out = model(input_ids, targets=y, parallel=True)
# or
out = model.forward_parallel(input_ids, targets=y)
```

## Structure

```text
X = Embed(tokens)                 # all T
D_win = DendriteWindow(X)         # causal
H = D_win
for j in 1..k:                    # depth, not time
    E = sg(X) - shift(Predictor(H))
    D = D_win + gamma * E
    O = RouterSkills(H)
    M,R = DeltaMemory over T
    H = FieldStep(H, D, shift(R))
Z = Readout(...)
```

Training / serving scripts keep the sequential default so nothing breaks.
