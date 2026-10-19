# Data

| Path | Role |
|------|------|
| `samples/` | Tiny demo corpora (identity, code, multilingual) |
| `align/` | Alignment / SFT style snippets |
| `raw/` | Optional: put your own large `.txt` dumps here |
| `tokens/` | **Generated** tokenized `.bin` (via `prepare_data.py`) — not required in git |

## Tokenize for training

```bash
# Your text files → train.bin / val.bin + tokenizer.json
python scripts/data/prepare_data.py \
  --input "data/raw/*.txt" "data/samples/*.txt" \
  --out-dir data/tokens \
  --vocab-size 32000
```

## Train

```bash
export PYTHONPATH=packages
python -m sentinel.cli train \
  --config configs/default.yaml \
  --data-dir data/tokens \
  --output-dir runs/local_run
```

Or with streaming for large corpora:

```bash
python -m sentinel.cli train \
  --data-dir data/tokens \
  --streaming \
  --batch-size 8 --seq-len 256 \
  --max-steps 10000 \
  --output-dir runs/exp1
```

See `docs/guides/TRAIN_1B.md` for large-scale data notes.
