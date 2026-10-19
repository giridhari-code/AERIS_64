# Decent-assistant corpus (1M+ tokens)

Put plain `.txt` / `.jsonl` files here. Target: **≥ 1,000,000 tokens** after BPE.

## Suggested mix
| Share | Content |
|------|---------|
| 40% | Natural dialogue (User: / Assistant:) English + Hinglish |
| 25% | Explanations / how-to / short essays |
| 15% | Code + comments |
| 10% | Identity + honesty (no chapluchi) lines |
| 10% | Domain you care about (security defense, product, …) |

## Bootstrap
Add real `.txt` / `.jsonl` under this folder until you have enough tokens
(rule of thumb: a few MB of clean text before BPE). Then:

```bash
python scripts/data/prepare_data.py --input "data/decent/*.txt" --out-dir data/tokens_decent
```

## Do not
- Repeat the same 14 lines 20k times (old AERIS_main failure mode)
- Train only on identity spam
