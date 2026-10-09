# Data

| Path | Role |
|------|------|
| `samples/` | Tiny demo corpora (identity, code, multilingual) |
| `align/` | Alignment / SFT style snippets |
| `ready/` | **Generated** train corpus (`build_dataset.py`) — not required in git |
| `raw/` | Optional: put your own large `.txt` dumps here |

## Build a train-ready dataset (local)

```bash
# ~100k synthetic chat + all samples/align → data/ready/train.txt
python scripts/data/build_dataset.py

# larger synthetic
python scripts/data/build_dataset.py --synthetic-chars 500000

# add your own files
python scripts/data/build_dataset.py --extra data/raw/my_books.txt data/raw/wiki.txt
```

## Train on it

```bash
export PYTHONPATH=src
python scripts/train/train_company.py \
  --data data/ready/train.txt \
  --tokenizer char \
  --preset small \
  --steps 2000 \
  --out checkpoints/local_run
```

## 1B-scale data (real)

`data/ready/train.txt` is **not** enough for 1B parameters.  
You need multi-GB clean text (billions of tokens). Put files under `data/raw/` and:

```bash
python scripts/data/build_dataset.py --synthetic-chars 0 --extra data/raw/*.txt
# or BPE pack:
python scripts/data/prepare_data.py --input "data/raw/*.txt" --out-dir data/tokens --vocab-size 32000
```

See `docs/guides/TRAIN_1B.md`.
