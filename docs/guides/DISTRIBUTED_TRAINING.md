# Distributed training (DDP)

Data-parallel training with **torchrun**. This is **multi-GPU / multi-process** data parallel.

## Single process (default)

```bash
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/ready/pretrain_raw.txt \
  --preset small --steps 1500 \
  --out docs/runs/pretrain_v1 --device cuda
```

## Multi-GPU

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 \
  scripts/train/train_company.py \
  --data data/ready/pretrain_raw.txt \
  --preset small --steps 1500 \
  --out docs/runs/ddp_v1 --device cuda
```

| Flag / env | Meaning |
|------------|---------|
| `--nproc_per_node=N` | N processes (usually N GPUs) |
| `LOCAL_RANK` | Set by torchrun; each process one GPU |
| Rank 0 | Logs, checkpoint, eval |

## Trainer path

`neurofield.training.trainer.Trainer` already wraps **DDP** when `init_distributed_if_needed()` runs (CLI train).

## Helpers

`src/neurofield/training/distributed.py`:

- `init_distributed_if_needed()`
- `is_main()`, `barrier()`, `all_reduce_mean()`
- `unwrap_model()`, `cleanup()`

## Notes

- Batch is **sharded by process** (each rank different random slices in `train_company`).
- Effective throughput ≈ `steps × world_size` token batches.
- NCCL needs NVIDIA GPUs; CPU multi-process uses gloo.
- Save **only rank 0** unwrapped weights (`model.safetensors`).
