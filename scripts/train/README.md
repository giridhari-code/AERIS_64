# Training (company standard)

## Single-stage LM

```bash
export PYTHONPATH=packages
python scripts/data/prepare_data.py --input "data/raw/*.txt" --out-dir data/tokens
python -m sentinel.cli train --config configs/default.yaml --data-dir data/tokens --output-dir runs/exp1
```

## Full multi-stage pipeline

| Stage | Purpose |
|-------|---------|
| `pretrain` | Next-token pretraining on token bins |
| `midtrain` | Continued pretrain, lower LR, higher-quality mix |
| `long_context` | Longer `seq_len`, context extension |
| `sft` | Supervised fine-tuning (instruction / chat) |
| `reasoning` | CoT / reasoning-trace SFT |
| `preference_bt` | Bradley-Terry reward model |
| `dpo` | Direct Preference Optimization |
| `orpo` | Odds-Ratio Preference Optimization |
| `lora_pref` | LoRA + DPO (PEFT preference) |
| `grpo` | Group Relative Policy Optimization |
| `rlhf_ppo` | On-policy PPO + KL to reference |
| `opd` | Online Preference Distillation |

```bash
python -m sentinel.cli pipeline --list-stages

python -m sentinel.cli pipeline \
  --config configs/colab.yaml \
  --stages sft,dpo,grpo \
  --sft-data data/align/sft_demos.txt \
  --prefs data/demo/preferences.jsonl \
  --output-dir runs/align

python -m sentinel.cli pipeline \
  --pipeline-config configs/pipeline/default.yaml \
  --config configs/default.yaml \
  --data-dir data/tokens \
  --sft-data data/align/sft_demos.txt \
  --prefs data/demo/preferences.jsonl \
  --output-dir runs/full_pipeline
```

Preference JSONL: `{"prompt":"...","chosen":"...","rejected":"..."}`

Artifacts: `runs/<exp>/<stage>/` (safetensors + checkpoint.pt).
