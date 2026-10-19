# Data layout (company)

| Path | Role |
|------|------|
| Your corpus | Raw text for `prepare_data.py` |
| Token output | `train.bin` / `val.bin` / `tokenizer.json` from `prepare_data.py` |
| SFT | Instruction JSONL or dialogue text — pass via `--sft-data` |
| Preferences | JSONL with `prompt` / `chosen` / `rejected` — pass via `--prefs` |

No default dataset is selected by the training code. All paths are explicit CLI/config inputs.

```bash
python scripts/data/prepare_data.py --input "/data/raw/*.txt" --out-dir /data/tokens --vocab-size 32000
python -m sentinel.cli train --data-dir /data/tokens --config configs/default.yaml --output-dir /runs/exp1
```
