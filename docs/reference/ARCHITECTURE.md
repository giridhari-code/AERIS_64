# NeuroField / AESC architecture

## Per-token loop (every word)

```text
for t in 0 .. T-1:
    x_t  → Embedding
         → Dendrite(x_hist, delayed_error)  → d_t   # changes every token
         → Metacognition(S, conflict)       → k, gate_bias
         → NeuralField(h, d_t, r, k)        → h     # k steps on this d_t
         → Predictor(h)                     → pred for x_{t+1}
         → Router / Skills(h)               → o_t
         → Neuromodulator(S, h, bias)       → g
         → FastMemory write/read            → M, r
         → SlowMemory read (ring slots)     → u
         → logits_t = Head(o_t + mem)
```

**Important:** `d_t` is recomputed on every token from the current dendrite
window and the delayed prediction error. The field does **not** reuse a
static context vector across words.

## Two-speed memory

| Path | Role | Lifetime |
|------|------|----------|
| **Fast M** | In-segment associations (delta rule) | ~segment_len tokens |
| **Slow ring** | Context window | n_slots × segment ≈ max_seq_len |

### Slow memory = context window

```text
n_slots ≈ max_seq_len / segment_len

At segment boundary (and sequence end):
    fast M  →  write into next ring slot   (overwrite oldest when full)

Read:
    query = W_q(d_t)
    score all filled slots → softmax → weighted read
    (+ small global prior M_s from training replay)
```

Long context is handled **here**, not by stretching fast M.

## Timing (no leakage)

- Predictor at step `t` forecasts embedding of `x_{t+1}`.
- Error is applied only when `x_{t+1}` arrives (step `t+1` onward).
- Logits for position `t` never see `x_{t+1}`.

## Modules

| Module | Output |
|--------|--------|
| Embedding | x_t |
| Dendrite | d_t |
| Metacognition | k, gate_bias, confidence |
| Neural field | h |
| Predictor | surprise S, error |
| Router + skills | o_t |
| Neuromodulator | write gate g |
| Fast memory | M, r |
| Slow memory | u (context slots) |
| Head | logits |

## Config knobs

- `max_seq_len` — context window target  
- `truncate_write_window` / `SafetyConfig.truncate_write_window` — segment length  
- `context_slots` — 0 = auto `ceil(max_seq_len / segment)`  
