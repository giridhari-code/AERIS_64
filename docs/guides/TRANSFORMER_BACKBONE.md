# Transformer backbone (default)

AERIS train/serve default is a **decoder-only Transformer** (GPT-style):

- token + positional embedding
- `n_layers` blocks: causal self-attention + MLP
- causal mask (no future tokens)
- optional weight-tied LM head
- label smoothing / dropout via train flags

```bash
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/ready/pretrain_raw.txt \
  --preset small --steps 800 \
  --label-smoothing 0.1 --dropout 0.1 \
  --out docs/runs/tf_v1 --device cuda
```

Legacy NeuroField modules remain under `src/neurofield/modules/` (dendrite, field, memory) for research; they are **not** the default `forward()` path.

Config keys: `architecture="transformer"`, `n_layers`, `n_heads`, `ffn_mult`.
