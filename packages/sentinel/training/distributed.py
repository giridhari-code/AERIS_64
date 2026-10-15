"""Data-parallel helpers (torchrun / DDP). Single-process runs are unaffected.

Launch examples
---------------
  # 1 GPU / CPU (no change)
  PYTHONPATH=src python scripts/train/train_company.py --data ... --out ...

  # 2+ GPUs
  PYTHONPATH=src torchrun --standalone --nproc_per_node=2 \\
    scripts/train/train_company.py --data ... --out ... --device cuda

  # module entry
  torchrun --nproc_per_node=4 -m sentinel.cli train --config ...
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist


def world_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from the torchrun environment."""
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def init_distributed_if_needed(timeout_min: int = 30) -> bool:
    """Join the process group when launched by torchrun (WORLD_SIZE > 1).

    Returns True if this process is part of a distributed job.
    """
    rank, world, local = world_info()
    if world <= 1 or not dist.is_available():
        return False
    if not dist.is_initialized():
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local)
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_min),
        )
        if is_main():
            print(
                f"[distributed] backend={backend} world_size={world} rank={rank} local_rank={local}"
            )
    return True


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """In-place mean over ranks; no-op if single process."""
    if not is_distributed() or get_world_size() == 1:
        return t
    x = t.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= get_world_size()
    return x


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable Python object from src rank."""
    if not is_distributed():
        return obj
    store: list[Any] = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(store, src=src)
    return store[0]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP wrapper if present."""
    return model.module if hasattr(model, "module") else model


def cleanup() -> None:
    if is_distributed():
        barrier()
        dist.destroy_process_group()
