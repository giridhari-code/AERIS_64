"""Data-parallel helpers (torchrun / DDP). Single-process runs are unaffected.

Launch:  torchrun --nproc_per_node=4 -m neurofield.cli train --config ...
"""

from __future__ import annotations

import os
from datetime import timedelta

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
    """Join the process group when launched by torchrun (WORLD_SIZE > 1). Returns True if distributed."""
    rank, world, local = world_info()
    if world <= 1 or not dist.is_available():
        return False
    if not dist.is_initialized():
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local)
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=timeout_min)
        )
    return True


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def cleanup() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()
