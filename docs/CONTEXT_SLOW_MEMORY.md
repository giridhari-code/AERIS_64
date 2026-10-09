# Context window in slow memory

## Design

| Path | Role |
|------|------|
| **Fast memory `M`** | Short segment (delta-rule within `truncate_write_window` tokens) |
| **Slow memory ring** | **Context window**: one slot per segment, `n_slots ≈ max_seq_len / segment_len` |

At every segment boundary, the current fast matrix is written into the next ring slot.
Read queries **all filled slots** (content-addressed) so long context is handled on the slow path.

```text
tokens ──► fast M (segment)
              │ every truncate_window steps
              ▼
         slow slots[0..n_slots-1]  ◄── context window
              │
              ▼
         u = attend(slots) + small global prior M_s
```

## Config

- `ModelConfig.context_slots` — ring size (0 = auto from `max_seq_len // truncate_write_window`)
- `SafetyConfig.truncate_write_window` — segment length (default 32)

## Code

- `SlowMemory.write_slot` / `read(..., slots=, filled=)`
- Model state: `slow_slots`, `slow_ptr`, `slow_filled`
- `W_q` tied to fast `W_k` at init
