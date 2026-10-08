# AERIS_64 improvements pack

| Request | What we added | Honest limit |
|---------|---------------|--------------|
| **Real intelligence** | Better train path (BPE default, resume, presets) | Still needs **more data + larger preset + time** — not a switch |
| **Stable short replies** | Auto length + trim first line + strong stop | Helps; not perfect on tiny models |
| **Strong stop** | Stop on `User:`, blank lines, sentence end, repeat | In `adaptive_length.py` |
| **BPE default** | `train_company.py --tokenizer bpe` default, char fallback | Retrain to use BPE checkpoint |
| **Continue from ckpt** | `--resume docs/AERIS_64` | Same vocab/size works best |
| **Eval suite daily** | `scripts/daily_eval.py` → `docs/logs/` | Run once per day |
| **Multi-turn memory** | Session text history + model state | Last ~6 turns; not long-term memory |

## Commands

```bash
# continue training
PYTHONPATH=src python scripts/train_company.py --resume docs/AERIS_64 --data data/my_english.txt --steps 500 --out docs/AERIS_64

# BPE (default now)
PYTHONPATH=src python scripts/train_company.py --tokenizer bpe --steps 500 --out docs/AERIS_64

# daily eval
PYTHONPATH=src python scripts/daily_eval.py --checkpoint docs/AERIS_64
```


## Anthropic-style mini align

See [ANTHROPIC_STYLE_ALIGN.md](ANTHROPIC_STYLE_ALIGN.md) and .

## Reinforcement learning
See [REINFORCEMENT_LEARNING.md](REINFORCEMENT_LEARNING.md).
