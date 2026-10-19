# data/demo — demo dataset (AERIS_64, ~1M params, char-level)

Training ki har stage ki apni file. Sab kuch **local** — koi internet nahi, koi coding
content nahi. Format `data/samples/` + `data/align/` jaisa hi hai, bas ek jagah, aur
whole pipeline isi folder se chalti hai.

| File | Stage | Format |
|------|-------|--------|
| `pretraining.txt` | 1. Pretraining | Plain text (identity, memory architecture, Hindi/Hinglish, daily talk) |
| `sft.txt` | 2. SFT (chat) | `User:` / `Assistant:` pairs + special-token examples |
| `sft_tools.txt` | 2b. SFT (tool calling) | `System:` / `User:` / `Assistant:` (JSON tool call) |
| `sft_reasoning.txt` | 2c. SFT (reasoning) | `User:` / `Assistant:` with `Answer:` |
| `constitution.txt` | 2d. Rules / persona | Numbered rules |
| `preferences.jsonl` | 3. Preference (DPO/RLHF) | JSON per line: `prompt` / `chosen` / `rejected` |
| `rl_prompts.txt` | 4. RL prompts | One prompt per line (`#` = comment) |
| `eval/tasks.jsonl` | 5. Eval (pass/fail) | JSON per line: `expect_contains_any` |
| `eval/redteam.jsonl` | 5. Eval (safety) | JSON per line: `expect_refuse` |

Niyam: training aur inference ka format bilkul same rakho (`System:` / `User:` /
`Assistant:`, special tokens). Eval data ko training mein kabhi mix mat karo.

## Run (repo root se)

```bash
export PYTHONPATH=packages

# 1) Pretraining — ~1M params, 1000 steps
python scripts/data/prepare_data.py --input data/demo/pretraining.txt --out-dir data/tokens_demo --vocab-size 512
PYTHONPATH=packages python -m sentinel.cli train --data-dir data/tokens_demo --output-dir runs/demo --device cpu --max-steps 1000

# 2) SFT + preference (aligned checkpoint)
# REMOVED — use: python -m sentinel.cli train \
  --resume checkpoints/demo_1m \
  --sft data/demo/sft.txt data/demo/sft_tools.txt data/demo/sft_reasoning.txt data/demo/constitution.txt \
  --prefs data/demo/preferences.jsonl \
  --preset demo_1m --out checkpoints/demo_1m_sft --device cpu

# 3) RLHF (reward -> dpo -> ppo)
# REMOVED — RLHF scripts deleted; use cli train for LM only \
  --resume checkpoints/demo_1m_sft \
  --prefs data/demo/preferences.jsonl \
  --prompts data/demo/rl_prompts.txt \
  --preset demo_1m --out checkpoints/demo_1m_rlhf --device cpu

# 4) Eval (checkpoint naapne ke liye — train nahi)
python scripts/eval/run_e2e_tasks.py --checkpoint checkpoints/demo_1m_sft \
  --tasks data/demo/eval/tasks.jsonl --out data/demo/eval/report.json

python scripts/eval/red_team.py --checkpoint checkpoints/demo_1m_sft \
  --tasks data/demo/eval/redteam.jsonl --out data/demo/eval/redteam_report.json

# 5) Param count verify
python scripts/research/count_params.py --checkpoint checkpoints/demo_1m
```

Preset `demo_1m` (`configs/presets.yaml`): `d_model=160`, `d_k=d_v=80`, `n_skills=8`
→ **~1.0M parameters** (char vocab ke saath).

## Data ki quality ke rules

1. Pretraining = plain text. SFT = sawaal-jawab. Preference = achha/bura pair.
2. Special tokens (`<|user|>` etc.) chalte hain — `User:` / `Assistant:` ko tokenizer
   khud alias karta hai. Dono formats ka sample `sft.txt` mein hai.
3. Tool calling mein tool waale aur bina tool waale, dono examples chahiye (warna
   model har baat par tool call karega).
4. Katai wala jawab bhi sikhaya gaya hai ("I don't know" / "I cannot read a live
   clock") — taaki model jhooth na banaye.
5. Eval data (tasks/redteam) ko training mein kabhi mix mat karo.
