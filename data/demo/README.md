# data/demo — demo dataset (AERIS_64)

Local-only demo files for the company pipeline. No internet required.

| File | Stage | Format |
|------|-------|--------|
| `pretraining.txt` | pretrain | Plain text |
| `sft.txt` | sft | `User:` / `Assistant:` |
| `sft_tools.txt` | sft | tool-calling dialogue |
| `sft_reasoning.txt` | reasoning | CoT-style `Answer:` |
| `constitution.txt` | sft (optional mix) | numbered rules |
| `preferences.jsonl` | preference_bt / dpo / orpo / grpo / rlhf_ppo | `prompt` / `chosen` / `rejected` |
| `rl_prompts.txt` | prompts list | one prompt per line |
| `eval/*.jsonl` | eval only | never mix into train |

## Run (repo root)

```bash
export PYTHONPATH=packages

python scripts/data/prepare_data.py \
  --input data/demo/pretraining.txt \
  --out-dir data/tokens_demo --vocab-size 512

python -m sentinel.cli train \
  --config configs/colab.yaml \
  --data-dir data/tokens_demo \
  --output-dir runs/demo_pretrain \
  --device cpu --precision fp32 --max-steps 200

python -m sentinel.cli pipeline \
  --config configs/colab.yaml \
  --stages sft,reasoning,preference_bt,dpo,grpo \
  --sft-data data/demo/sft.txt \
  --reasoning-data data/demo/sft_reasoning.txt \
  --prefs data/demo/preferences.jsonl \
  --output-dir runs/demo_align \
  --device cpu
```

See `scripts/train/README.md` for the full stage list.
