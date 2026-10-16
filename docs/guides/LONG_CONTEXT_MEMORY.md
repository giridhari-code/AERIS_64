# Long context + memory (SENTINEL)

Code: `packages/sentinel/serving/chat_history.py`, `packages/sentinel/modules/memory.py`

## Two systems

| Layer | What | Bound |
|-------|------|--------|
| **Fast memory M** | Short working segment (delta-rule) | Fixed (B, d_k, d_v); segment ≈ `truncate_write_window` (32) |
| **Slow memory ring** | Long context snapshots | `n_slots ≈ max_seq_len / segment` |
| **Chat history text** | Multi-turn prompt re-feed | `NEUROFIELD_MAX_TURNS` (default **24**) |
| **Token truncate** | Encode safety | `NEUROFIELD_MAX_CONTEXT` / `max_seq_len` (default 2048) |
| **User memory** | Disk slow ring + transcript | `UserMemoryStore` |

## Env

```bash
export NEUROFIELD_MAX_TURNS=32          # chat pairs kept in RAM prompt
export NEUROFIELD_TURN_CHARS=2000       # chars per turn stored
export NEUROFIELD_MAX_CONTEXT=2048      # max tokens into model
```

## Fix applied (product)

- Default chat turns **6 → 24** (env-configurable up to 128)
- Per-turn store **500 → 1500** chars
- Server logs long-context limits on model load

## Honest limit

Neural slow ring quality still depends on **training**. Longer chat window only helps **text** continuity; weak checkpoints still produce weak text.


## Commit trail

- `82eec15` — 6→24 turns, env limits
- follow-up — public `get_max_turns()`, packages/ path
