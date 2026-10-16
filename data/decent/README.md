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

## Bootstrap (small, not 1M)
```bash
python scripts/data/make_synthetic_100k.py \
  --out data/decent/synthetic_seed.txt --target-chars 200000
```
Then **add real text** until `wc -c data/decent/*.txt` is roughly 4–8 MB+ (rule of thumb before BPE).

## Do not
- Repeat the same 14 lines 20k times (old AERIS_main failure mode)
- Train only on identity spam
