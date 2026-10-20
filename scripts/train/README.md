# Training (company standard)

**Product default: AERIS-109M (~108.15M params)** — `configs/aeris_109m.yaml` / `configs/default.yaml`

## Dependencies

```bash
pip install -e .
# safetensors + tokenizers included
```

## 1) Tokenize (vocab must match config: 32000)

```bash
export PYTHONPATH=packages
python scripts/data/prepare_data.py \
  --input "/data/raw/*.txt" \
  --out-dir /data/tokens \
  --vocab-size 32000 \
  --val-frac 0.01
```

## 2) Pretrain 109M

```bash
python -m sentinel.cli train \
  --config configs/aeris_109m.yaml \
  --train-data /data/tokens/train.bin \
  --val-data /data/tokens/val.bin \
  --output-dir runs/aeris_109m \
  --precision bf16 --device cuda
```

Effective batch ≈ `batch_size × grad_accum` (4×4=16). OOM → lower `seq_len` / batch.

## 3) Pipeline stages

```bash
python -m sentinel.cli pipeline \
  --config configs/aeris_109m.yaml \
  --stages sft,dpo \
  --sft-data /data/sft.jsonl \
  --prefs /data/preferences.jsonl \
  --tokenizer-json /data/tokens/tokenizer.json \
  --output-dir runs/align_109m
```

## Optional small smoke only

`configs/colab.yaml` — optional small GPU smoke only (not product default).

## Scale notes

| Config | Params | Use |
|--------|--------|-----|
| aeris_109m / default | ~108M | Architecture proof + product path |
| colab | small | Free Colab smoke only |
| Future 3.4B | — | Multi-GPU + billions of tokens |
