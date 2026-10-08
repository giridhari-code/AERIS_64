# Reinforcement learning / RLHF (full pipeline)

## Stack

| Stage | Script / module | Role |
|-------|-----------------|------|
| Reward model | `scripts/train_rlhf.py --stages reward` | Bradley-Terry pairwise ranking on sequence log-probs |
| DPO | `--stages dpo` | Direct Preference Optimization vs frozen reference |
| PPO | `--stages ppo` | On-policy sample + RM score + clipped objective + KL to ref |
| Libraries | `src/neurofield/rlhf/` | `rewards.py`, `dpo.py`, `ppo.py` |

Legacy single-file REINFORCE: `scripts/train_rl.py` (still available).

## Run full stack

```bash
# 0) supervised base first
PYTHONPATH=src python scripts/train_company.py \
  --data data/align/sft_demos.txt data/rl_prompts.txt \
  --steps 300 --out docs/AERIS_64

# 1) full RLHF-style: reward + DPO + PPO
PYTHONPATH=src python scripts/train_rlhf.py \
  --resume docs/AERIS_64 \
  --prefs data/align/preferences.jsonl \
  --prompts data/rl_prompts.txt \
  --stages reward,dpo,ppo \
  --reward-steps 150 \
  --dpo-steps 150 \
  --ppo-steps 100 \
  --out docs/AERIS_64
```

## Single stage

```bash
PYTHONPATH=src python scripts/train_rlhf.py --stages dpo --dpo-steps 200 --out docs/AERIS_64
```

## Preference data format

`data/align/preferences.jsonl`:

```json
{"prompt":"hello","chosen":"Hello. I am AERIS_64.","rejected":"garbage text"}
```

## Honest limits

- **Algorithmically complete** RM → DPO → PPO path for this codebase.
- **Not** multi-GPU, not large reward transformers, not Anthropic/OpenAI scale.
- Quality still needs **good preference data** and a stronger base model.
