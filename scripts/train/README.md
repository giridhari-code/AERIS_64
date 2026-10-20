# Training (company standard)

No demo paths. Pass your own data and tokenizer.

## Dependencies

```bash
pip install -e .          # includes safetensors + tokenizers
# or: pip install safetensors tokenizers
```

**Colab:** always do both:

```python
import sys, os
PKG = "/content/AERIS_64/packages"
sys.path.insert(0, PKG)
os.environ["PYTHONPATH"] = PKG
```

## 1) Tokenize

```bash
export PYTHONPATH=packages
python scripts/data/prepare_data.py \
  --input "/data/corpus/*.txt" \
  --out-dir /data/tokens \
  --vocab-size 8000 \
  --val-frac 0.05
```

Tiny text: val split is auto-filled if hash split yields zero val chunks.

## 2) Single-stage LM

```bash
python -m sentinel.cli train \
  --config configs/aeris_2m.yaml \
  --train-data /data/tokens/train.bin \
  --val-data /data/tokens/val.bin \
  --output-dir /runs/pretrain \
  --precision bf16
```

Small bins automatically enable **streaming** (safer on CPU / low RAM).

## 3) Multi-stage pipeline

```bash
python -m sentinel.cli pipeline --list-stages

python -m sentinel.cli pipeline \
  --config configs/aeris_2m.yaml \
  --stages sft,dpo \
  --sft-data /data/sft.jsonl \
  --prefs /data/preferences.jsonl \
  --tokenizer-json /data/tokens/tokenizer.json \
  --resume /runs/pretrain \
  --output-dir /runs/align
```

`--resume` accepts: `checkpoint_*.pt`, `model.safetensors`, or an export **folder**.
CLI `--stages` overrides pipeline YAML stages.

Stage folders save: `model.safetensors` + `checkpoint.pt` + `checkpoint_final.pt` + `checkpoint_latest.pt`.

## Quality expectations (not bugs)

| Data / steps | Expected |
|--------------|----------|
| ~1k–30k words, few hundred steps | High CE, weak/garbage text |
| Pipeline stages all green | Means code ran — **not** that the LM is smart |
| Usable small product | Much more clean text + thousands of steps |
| 1B-class base | Multi-GPU + **billions** of tokens — not Colab free |

## Configs

| File | Use |
|------|-----|
| `configs/colab.yaml` | T4 / ~1–2M |
| `configs/aeris_2m.yaml` | ~2.25M product preset |
| `configs/default.yaml` | Larger defaults |
| `configs/pipeline/default.yaml` | Stage schedule |
