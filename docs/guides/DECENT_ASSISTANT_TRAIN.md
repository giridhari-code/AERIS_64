# Decent assistant training (1M+ tokens · d_model 512)

## Honest requirements

| Need | Why |
|------|-----|
| **GPU** (8GB+ VRAM better) | 512-d + seq 512 on CPU = days/weeks |
| **≥1M tokens** real-ish text | Old AERIS_main (~300 tokens / 14 unique) → `cc cc` garbage |
| **BPE vocab ≥2k–4k** | Char vocab 44 cannot write open language |
| **Val split + no dup spam** | Prevent train NLL→0, val collapse |

Preset **`large`** in `configs/presets.yaml` is already **d_model=512**.

## Pipeline

```bash
# 1) Data
mkdir -p data/decent
# add many .txt files OR:
python scripts/data/make_synthetic_100k.py \
  --out data/decent/synthetic_seed.txt --target-chars 500000
# then ADD real licensed text until ~1M+ tokens

# 2) Train (GPU)
export NEUROFIELD_DEVICE=cuda
./scripts/train/run_decent_assistant.sh

# 3) Serve
CHECKPOINT=checkpoints/AERIS_decent make backend
```

## CPU (smoke only)

```bash
SMOKE_STEPS=100 NEUROFIELD_DEVICE=cpu ./scripts/train/run_decent_assistant.sh
```

## After train — quality checklist

- [ ] `who are you?` → stable AERIS / SENTINEL identity  
- [ ] No letter spam (`cc cc…`)  
- [ ] Short honest “I don’t know” when needed  
- [ ] Val loss not exploding vs train  

## Bigger than this

`configs/aeris_1b.yaml` + `docs/TRAIN_1B.md` — different scale (cluster).
