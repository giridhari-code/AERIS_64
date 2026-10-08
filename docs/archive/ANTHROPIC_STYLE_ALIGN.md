# Anthropic-style pipeline on AERIS_64 (mini)

Maps **public Claude training ideas** onto a **toy** NeuroField model.

| Anthropic stage | AERIS command / files |
|-----------------|------------------------|
| Pre-training | `scripts/train_company.py` on your `.txt` data |
| SFT (demos) | `data/align/sft_demos.txt` via `scripts/train_align.py` |
| Constitution | `data/align/constitution.txt` (principles as text) |
| Preference / RLAIF-like | `data/align/preferences.jsonl` — simple chosen vs rejected loss |
| Full CAI critique loops + PPO | **Not implemented** (needs large model + infra) |

## Run order

```bash
# 1) base train (English etc.)
PYTHONPATH=src python scripts/train_company.py \
  --data data/my_english.txt --tokenizer char --steps 500 --out docs/AERIS_64

# 2) mini align (SFT + preferences)
PYTHONPATH=src python scripts/train_align.py \
  --resume docs/AERIS_64 \
  --steps-sft 200 \
  --steps-pref 100 \
  --out docs/AERIS_64
```

## Honest limits

- This does **not** make Claude.
- Constitution is **read as training text**, not a full self-critique RLAIF stack.
- Preference step is a **simple NLL hinge**, not PPO reward model training.
- Quality still scales with **data + size + time**.

## Edit your constitution / prefs

1. Edit `data/align/constitution.txt`
2. Add demos to `data/align/sft_demos.txt`
3. Add JSON lines to `data/align/preferences.jsonl`:
   `{"prompt":"...", "chosen":"good reply", "rejected":"bad reply"}`
4. Re-run `train_align.py`
