# AERIS-42M — first pretrain dataset

**Target model:** ~42M params (`--preset aeris_42m`)  
**Languages:** English · Hindi · Bangla (Bengali)  
**No code** in this corpus.

## Size targets (first pretrain)

| Level | Words (approx) | Tokens (approx) |
|-------|----------------|-----------------|
| Smoke / pipeline test | 0.5–2M | ~1–3M |
| **Minimum useful** | **50–100M** | ~70–150M |
| **Recommended first full** | **200–250M** | ~250–400M |

Balance roughly **1/3 each** language by tokens when possible.

## Folder layout

```text
data/pretrain_42m/
  en/          # English .txt / .jsonl
  hi/          # Hindi (Devanagari)
  bn/          # Bangla
  mixed/       # optional multilingual dialogues
  SOURCES.md   # licence notes for every real source you add
  seed/        # small synthetic bootstrap only
```

## Build steps

```bash
# 1) Small seed (NOT enough alone for quality)
python scripts/data/build_pretrain_42m_seed.py --target-chars-per-lang 150000

# 2) Add REAL text under en/ hi/ bn/ (public domain, your writing, clear licence)
#    No commercial ebooks, no pirated dumps — see docs/guides/COPYRIGHT_POLICY.md

# 3) Stats
python scripts/data/stats_pretrain_42m.py

# 4) Train when data is large enough
PYTHONPATH=packages python scripts/train/train_company.py \
  --preset aeris_42m --tokenizer bpe --bpe-vocab 32000 \
  --data data/pretrain_42m/en data/pretrain_42m/hi data/pretrain_42m/bn data/pretrain_42m/mixed data/pretrain_42m/seed \
  --steps 20000 --device cuda --bf16 --out checkpoints/AERIS_42m
```

If `train_company` needs file paths not directories, expand:

```bash
find data/pretrain_42m -name '*.txt' -type f | sort > /tmp/42m_files.txt
# pass those paths to --data
```

## Do not

- Code corpora (Python/JS dumps, GitHub bulk)
- Full copyrighted books
- Same 20 lines repeated millions of times
