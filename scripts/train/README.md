# Training (company standard)

**One entry point only.** Do not add parallel training loops.

```text
raw text
  → scripts/data/prepare_data.py     # tokenize → train.bin / val.bin
  → python -m sentinel.cli train     # packages/sentinel/cli.py
       → data/dataset.py             # DataLoader
       → model.py + modules/         # forward / loss
       → training/trainer.py         # optim, EMA, ckpt, eval, DDP
  → runs/<exp>/checkpoint_*.pt + safetensors export
```

## Commands

```bash
export PYTHONPATH=packages

# 1) Tokenize
python scripts/data/prepare_data.py \
  --input "data/raw/*.txt" \
  --out-dir data/tokens \
  --vocab-size 32000

# 2) Train
python -m sentinel.cli train \
  --config configs/default.yaml \
  --data-dir data/tokens \
  --output-dir runs/exp1 \
  --precision bf16

# Large corpus
python -m sentinel.cli train \
  --data-dir data/tokens \
  --streaming \
  --batch-size 8 --seq-len 256 \
  --max-steps 50000 \
  --output-dir runs/exp1

# Multi-GPU
torchrun --nproc_per_node=2 -m sentinel.cli train \
  --data-dir data/tokens --streaming --precision bf16 \
  --output-dir runs/exp1

# Resume
python -m sentinel.cli train \
  --data-dir data/tokens \
  --output-dir runs/exp1 \
  --resume
```

## Equivalent

```bash
python scripts/train/train.py   # same as python -m sentinel.cli train
# or after pip install -e .
sentinel-train --data-dir data/tokens --config configs/default.yaml
```

## What lives where

| Path | Role |
|------|------|
| `packages/sentinel/cli.py` | CLI flags + wiring |
| `packages/sentinel/training/trainer.py` | Training loop |
| `packages/sentinel/data/dataset.py` | Token loaders |
| `scripts/data/prepare_data.py` | Text → binary tokens |
| `configs/default.yaml` | Defaults |

Research / RL / SFT experiments belong under `scripts/research/` or a future dedicated package — **not** a second `train_*.py` loop under `scripts/train/`.
