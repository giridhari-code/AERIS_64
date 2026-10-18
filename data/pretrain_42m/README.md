# AERIS-42M — first pretrain dataset

**Only real, high-quality text.** No synthetic bootstrap corpora.

| | |
|--|--|
| Model | `--preset aeris_42m` (~42M params) |
| Languages | English · Hindi · Bangla |
| Code | **None** — do not add source-code dumps |
| Synthetic | **Forbidden** for this recipe |

## Size targets

| Level | Words (approx) | Notes |
|-------|----------------|--------|
| Smoke (pipeline only) | small sample of *real* text | still real, not generated templates |
| Minimum useful | **50–100M** | |
| Recommended first full | **200–250M+** | balanced EN / HI / BN |

## Folders

```text
data/pretrain_42m/
  en/         # English — real prose, dialogue, articles you may use
  hi/         # Hindi (Devanagari)
  bn/         # Bangla
  mixed/      # optional real multilingual
  SOURCES.md  # required: every source + licence
```

Put **`.txt` / `.jsonl` / `.md`** files only from sources you document in `SOURCES.md`.

## What counts as high quality

| Use | Avoid |
|-----|--------|
| Clean books/articles with a **clear licence** (public domain, CC that allows ML, or your own writing) | Pirated ebooks / shadow libraries |
| Well-edited news or explainers you have rights to use | Random low-quality scrape spam |
| Real human Q&A / chat you own or licensed | Template “User:/Assistant:” spam loops |
| Educational text under an open licence | Full commercial books without rights |
| Balanced topics across the three languages | Code repos, stacktraces, minified JS |

Follow `docs/guides/COPYRIGHT_POLICY.md`.

## Stats

```bash
python scripts/data/stats_pretrain_42m.py
```

## Train (when real data is large enough)

```bash
PYTHONPATH=packages python scripts/train/train_company.py \
  --preset aeris_42m \
  --tokenizer bpe --bpe-vocab 32000 \
  --data data/pretrain_42m/en data/pretrain_42m/hi data/pretrain_42m/bn \
  --steps 20000 --batch-size 8 --seq-len 512 --grad-accum 4 \
  --lr 3e-4 --warmup 500 --cosine \
  --device cuda --bf16 \
  --out checkpoints/AERIS_42m --product AERIS_42m
```

Directories are expanded automatically (all `.txt` under each folder).

## Do not

- Run `build_pretrain_42m_seed.py` for the main corpus (legacy seed script is not part of this recipe)
- Mix synthetic identity spam as the bulk of training
- Add programming code corpora
