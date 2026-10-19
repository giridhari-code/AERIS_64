# Scripts

| Folder | Purpose |
|--------|---------|
| `train/` | Single alias → `python -m sentinel.cli train` (see `train/README.md`) |
| `data/` | `prepare_data.py` — tokenize text → `train.bin` / `val.bin` |
| `eval/` | Evaluation helpers |
| `research/` | Ablations, probes, paper experiments |
| `bench/` | Benchmarks |
| `ops/` | Ops utilities |

**Company training path:** `scripts/train/README.md`

All commands assume repo root and `PYTHONPATH=packages`.
