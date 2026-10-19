# Training (company standard)

No demo paths. Pass your own data and tokenizer.

## 1) Tokenize

```bash
export PYTHONPATH=packages
python scripts/data/prepare_data.py \
  --input "/data/corpus/*.txt" \
  --out-dir /data/tokens \
  --vocab-size 32000
```

## 2) Single-stage LM

```bash
python -m sentinel.cli train \
  --config configs/default.yaml \
  --data-dir /data/tokens \
  --output-dir /runs/pretrain \
  --precision bf16
```

## 3) Multi-stage pipeline

| Stage | Data flag |
|-------|-----------|
| pretrain / midtrain / long_context | `--data-dir` (train.bin) |
| sft | `--sft-data` |
| reasoning | `--reasoning-data` |
| preference_bt / dpo / orpo / lora_pref / grpo / rlhf_ppo | `--prefs` |
| opd | `--prefs` or `--sft-data` |

```bash
python -m sentinel.cli pipeline --list-stages

python -m sentinel.cli pipeline \
  --pipeline-config configs/pipeline/default.yaml \
  --config configs/default.yaml \
  --data-dir /data/tokens \
  --tokenizer-json /data/tokens/tokenizer.json \
  --sft-data /data/sft.jsonl \
  --reasoning-data /data/reasoning.jsonl \
  --prefs /data/preferences.jsonl \
  --output-dir /runs/pipeline
```

Preference JSONL schema: `{"prompt":"...","chosen":"...","rejected":"..."}`

SFT: `.jsonl` (`text` or instruction/output) or plain `.txt` chunks.

Artifacts: `/runs/pipeline/<stage>/`
