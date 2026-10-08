# Frontier-style scale & optimization (AERIS)

Frontier labs do not jump straight to 1B on a laptop. They **scale in stages**: data → tokens → width → schedule → align.

## Stage ladder

| Stage | Preset | Focus | Hardware |
|-------|--------|-------|----------|
| 0 | `tiny` | Pipeline works | CPU |
| 1 | `small` | Stable short chat | CPU / 1 GPU |
| 2 | `medium` | More capacity | 1 GPU better |
| 3 | `large` | Serious local scale | GPU |
| 4 | `xl` / `aeris_1b.yaml` | ~1B recipe | Multi-GPU / big VRAM |

```bash
# Stage 1 example — optimize schedule + accum
PYTHONPATH=src python scripts/train_company.py \
  --preset small \
  --data data/align/sft_demos.txt data/rl_prompts.txt data/company_corpus.txt \
  --tokenizer char \
  --steps 2000 \
  --batch-size 8 \
  --grad-accum 4 \
  --warmup 100 \
  --cosine \
  --lr 1e-3 \
  --out docs/AERIS_small
```

```bash
# Stage 2
PYTHONPATH=src python scripts/train_company.py \
  --preset medium --steps 5000 --grad-accum 8 --warmup 200 --cosine \
  --resume docs/AERIS_small --out docs/AERIS_medium
```

```bash
# Align after scale
PYTHONPATH=src python scripts/train_rlhf.py \
  --resume docs/AERIS_medium --stages reward,dpo,ppo --out docs/AERIS_medium
```

## Optimizations now in `train_company.py`

| Flag | Role |
|------|------|
| `--grad-accum N` | Larger effective batch without OOM |
| `--warmup N` | Linear LR warmup |
| `--cosine` | Cosine decay after warmup |
| `--weight-decay` | AdamW decay |
| `--bf16` | CUDA bfloat16 autocast |
| `--preset large\|xl` | Wider models |

## Frontier habits (do these)

1. **More clean data before more width**
2. **Eval every stage** (`daily_eval.py` / chat tests)
3. **Don’t max_tokens 8k on tiny models**
4. **Document each run** (steps, preset, data list)
5. **1B only with GPU + real corpus** (`docs/TRAIN_1B.md`)

## What “frontier-like” is not

- Not a switch named `frontier=true`
- Not 500 CPU steps on 10 KB text
- Not max_tokens alone

Scale = **data × params × tokens trained × good schedule × align**.
