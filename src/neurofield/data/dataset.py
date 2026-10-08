"""Data loading utilities – production, no hardcoded paths or toy demos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


class TokenDataset(Dataset):
    """
    Random-access dataset over a memory-mapped token array.
    Expects a .npy or .bin file of uint16 / int32 token ids.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        dtype: str = "uint16",
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        self.seq_len = seq_len
        if path.suffix == ".npy":
            self.data = np.load(path, mmap_mode="r")
        else:
            self.data = np.memmap(path, dtype=dtype, mode="r")
        self.n_tokens = len(self.data)
        if self.n_tokens < seq_len + 1:
            raise ValueError(
                f"File too short ({self.n_tokens} tokens) for seq_len={seq_len}"
            )

    def __len__(self) -> int:
        return self.n_tokens - self.seq_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return {"input_ids": x, "targets": y}


class StreamingTokenDataset(IterableDataset):
    """Infinite stream of random windows – useful for very large corpora."""

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        dtype: str = "uint16",
        seed: int = 42,
    ):
        path = Path(path)
        if path.suffix == ".npy":
            self.data = np.load(path, mmap_mode="r")
        else:
            self.data = np.memmap(path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # distinct stream per DataLoader worker (all workers used the same seed => duplicate batches)
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        while True:
            idx = rng.integers(0, self.n_tokens - self.seq_len)
            chunk = self.data[idx : idx + self.seq_len + 1]
            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))
            yield {"input_ids": x, "targets": y}


def build_dataloader(
    path: Optional[str],
    seq_len: int,
    batch_size: int,
    streaming: bool = False,
    num_workers: int = 0,
    dtype: str = "uint16",
) -> Optional[DataLoader]:
    if path is None:
        return None
    if streaming:
        ds = StreamingTokenDataset(path, seq_len, dtype=dtype)
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    ds = TokenDataset(path, seq_len, dtype=dtype)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
