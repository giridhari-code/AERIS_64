# Dynamic length + temperature (Grok-style)

When **auto** is on (`auto_length: true` in API / checkbox in UI):

| Prompt type | max tokens (budget) | temperature |
|-------------|---------------------|-------------|
| Short (`hi`, `thanks`) | ~48 | ~0.15 |
| Normal | ~96 | ~0.25 |
| Medium | ~256 | ~0.35 |
| Long / summarize / code | ~1024–2048 | ~0.40 |
| Creative (story, poem) | ~512 | ~0.70 |

UI **max tokens** = **cap** (upper limit), not fixed length.  
UI **temp** = soft blend with the dynamic value (35% slider when auto on).

Turn auto **off** → fixed slider values (old behavior).

Server: `adaptive_sampling()` in `src/neurofield/serving/adaptive_length.py`.
