# Training speed (AESC / NeuroField)

## Why it feels slow

The model is **recurrent over tokens**. For sequence length `T`, each example does `T` serial steps. A transformer of similar size is much faster because positions run in parallel.

**What is already parallel:** the **batch** axis — 64 sequences process together on GPU.

## Speed mode

```bash
python scripts/train/train_company.py \
  --data data/ready/train.txt \
  --tokenizer char \
  --preset small \
  --steps 1000 \
  --fast \
  --device auto \
  --out checkpoints/fast_run
```

`--fast` sets roughly: `seq_len=32`, `batch_size=64`, `compile=True`, `cosine=True`.

Rough effect vs `seq_len=128, batch=32`: often **several times** fewer serial steps per update.

## Honest limits

| Lever | Effect |
|-------|--------|
| Shorter `--seq-len` | Best single-knob speedup |
| Larger `--batch-size` | Better GPU use |
| `--compile` / bf16 / TF32 | Extra GPU efficiency |
| Multi-GPU DDP | Not free for this custom step (future) |

There is no free lunch that keeps long context and transformer-like wall-clock without changing the architecture.
