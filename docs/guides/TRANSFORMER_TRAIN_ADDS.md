# Transformer paper training adds (Attention Is All You Need §5)

Ported ideas that help AERIS without copying the whole MT setup.

| Paper | AERIS flag / behaviour |
|-------|-------------------------|
| Label smoothing ε=0.1 | `--label-smoothing 0.1` (default on in `train_company`) |
| Residual dropout 0.1 | `--dropout 0.1` |
| Warmup + decay | `--warmup N` + `--lr-schedule cosine\|noam\|constant` |
| Noam LR (§5.3) | `--lr-schedule noam` → `d^{-0.5} min(step^{-0.5}, step·warmup^{-1.5})` × `--lr` scale |

Example:

```bash
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/ready/pretrain_raw.txt \
  --preset small --steps 2000 \
  --label-smoothing 0.1 --dropout 0.1 \
  --lr-schedule noam --warmup 400 --lr 1.0 \
  --out docs/runs/noam_v1 --device cuda
```

Cosine (previous default style):

```bash
  --lr-schedule cosine --warmup 100 --lr 3e-3 --cosine
```

Not ported (different task): WMT BLEU, 25k-token length batching, 8×P100 schedule.
