# Reasoning — data + eval scores

## Goal

Measure **accuracy %** on fixed items. Do not claim reasoning without this number.

## Data (SFT)

File: `data/samples/sft_reasoning.txt`

Format:
```text
User: ...
Assistant: ... Answer: <final>
```

Expand to hundreds of pairs (math word problems, simple logic).

Train (after `git pull`, package `sentinel`):
```bash
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/samples/sft_reasoning.txt \
  --preset tiny --steps 500 \
  --out docs/runs/reason_sft_v1 --device cpu
```
(Use your current train entrypoint if renamed to `sentinel-train`.)

## Eval (tool-augmented — SI_v1)

```bash
PYTHONPATH=src python scripts/eval/reasoning_score.py
```

Report: `docs/audits/reasoning_score_report.json`

This suite scores **agent + calculator** (reliable path).  
Pure LM reasoning needs a separate model-only suite after strong SFT.

## Score interpretation

| Accuracy | Meaning |
|----------|---------|
| ~100% tool suite | SI_v1 tools OK |
| Low model-only | Need more SFT data |
| High both | Real progress |

## Next

1. Grow `sft_reasoning.txt` to 200–1000 pairs  
2. Train + model-only eval  
3. Word problems without calc tool
