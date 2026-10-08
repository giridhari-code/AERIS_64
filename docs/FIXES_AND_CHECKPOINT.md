# NeuroField v2 — Fixes, Safetensors & Multi-File Guide

## 1. Problem: `Error: model not loaded`

**Cause:** Server started without a checkpoint path.

**Fix:** Always set `CHECKPOINT`:

```bash
# Preferred (industry format — folder)
CHECKPOINT=docs/neurofield_trained PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# Legacy still works
CHECKPOINT=docs/neurofield_trained.pt PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Check: open `http://127.0.0.1:8000/readyz` → should say `"status": "ready"`.

---

## 2. Problem: Multiple confusing checkpoint files

**Before:** Several loose files:

- `docs/neurofield_trained.pt`
- `docs/neurofield_ddia_trained.pt`
- random train logs

Hard to know which file is “the model”.

**After (industry standard):** **One folder per model**

```
docs/neurofield_trained/
  model.safetensors   ← weights (safe, no pickle)
  config.json         ← architecture
  vocab.json          ← character vocab (chat)
  meta.json           ← training metadata
```

Point `CHECKPOINT` at the **folder**, not at many files.

| Checkpoint | Use |
|------------|-----|
| `docs/neurofield_trained/` | Identity + code demo |
| `docs/neurofield_ddia_trained/` | DDIA book snippets |
| `docs/neurofield_skills/` | After you train on `data/skills_real.txt` |

Legacy `.pt` files still load for compatibility, but **prefer the folder**.

---

## 3. Why safetensors? (industry standard)

| Format | Risk | Used by |
|--------|------|---------|
| `.pt` / pickle | Can execute arbitrary code on load | Old PyTorch only |
| **`.safetensors`** | Weights only, safe | Hugging Face, Diffusers, vLLM, industry |

NeuroField now saves/loads safetensors by default via `neurofield.checkpoint`.

---

## 4. Paper-faithful code fixes (already in this zip)

1. Fast memory value: `v = W_v(o_t)` (not `o_t + d_t`)
2. Slow replay uses **per-sequence surprise**
3. `bar_S` is **per-session** (no cross-user leak)
4. `max_write_norm` enforced; anomaly **freeze** wired in server
5. `param_report` counts **unique** parameters (tied embed/head once)

---

## 5. Train skills data → safetensors folder

```bash
cd neurofield_v2_clean
pip install -e ".[serve]"

PYTHONPATH=src python scripts/train_skills.py \
  --data data/skills_real.txt \
  --steps 800 \
  --out docs/neurofield_skills

CHECKPOINT=docs/neurofield_skills PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

---

## 6. Convert your own legacy `.pt` → safetensors folder

```python
from neurofield.checkpoint.io import convert_pt_to_safetensors
convert_pt_to_safetensors("path/to/old.pt", "docs/my_model")
```

If tied weights error appears, the converter clones shared tensors automatically.

---

## 7. Frontend

- Chat UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Needs checkpoint with `vocab.json` (or legacy `stoi`/`itos` in `.pt`) for `/v1/chat`

---

## 8. Quick health checklist

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"who am I","max_new_tokens":40,"temperature":0.3}'
```
