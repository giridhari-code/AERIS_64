"""Data loading utilities – production, no hardcoded paths or toy demos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
PathSpec = Union[PathLike, Sequence[PathLike]]


def _resolve_token_files(path: PathSpec) -> list[Path]:
    """Expand a file, directory, or list into concrete .bin/.npy paths."""
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]

    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            found = sorted(p.glob("*.bin")) + sorted(p.glob("*.npy"))
            # prefer train*.bin when present, but keep all token files
            if not found:
                raise FileNotFoundError(f"No .bin/.npy token files under directory: {p}")
            out.extend(found)
        elif p.exists():
            out.append(p)
        else:
            # glob pattern
            matches = sorted(Path().glob(str(p))) if any(c in str(p) for c in "*?[") else []
            if not matches:
                raise FileNotFoundError(f"Data file not found: {p}")
            out.extend(matches)
    # de-dupe while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for f in out:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _open_mmap(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=dtype, mode="r")


def _infer_dtype(path: Path, fallback: str = "uint16") -> str:
    """Best-effort dtype from sidecar or filename; else fallback."""
    side = path.with_suffix(path.suffix + ".dtype")
    if side.exists():
        text = side.read_text(encoding="utf-8").strip().lower()
        if text in ("uint16", "int32", "int64", "uint32"):
            return text
    name = path.name.lower()
    if "int32" in name or "i32" in name:
        return "int32"
    if "int64" in name or "i64" in name:
        return "int64"
    return fallback


class TokenDataset(Dataset):
    """
    Random-access dataset over a memory-mapped token array.

    Expects a .npy or .bin file of uint16 / int32 token ids.
    ``stride`` controls window spacing: 1 = max overlap (default),
    ``seq_len`` = non-overlapping packs (less correlated, fewer samples).
    """

    def __init__(
        self,
        path: PathLike,
        seq_len: int,
        dtype: str = "uint16",
        stride: int = 1,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        self.path = path
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.dtype = _infer_dtype(path, dtype) if dtype == "auto" else dtype
        self.data = _open_mmap(path, self.dtype)
        self.n_tokens = int(len(self.data))
        if self.n_tokens < self.seq_len + 1:
            raise ValueError(
                f"File too short ({self.n_tokens} tokens) for seq_len={self.seq_len}: {path}. "
                f"Lower --seq-len (e.g. 64 or 128) or add more tokenized data."
            )

    def __len__(self) -> int:
        # number of valid start positions with given stride
        return max(0, (self.n_tokens - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(idx) * self.stride
        chunk = self.data[start : start + self.seq_len + 1]
        x = torch.from_numpy(np.asarray(chunk[:-1], dtype=np.int64))
        y = torch.from_numpy(np.asarray(chunk[1:], dtype=np.int64))
        return {"input_ids": x, "targets": y}


class MultiFileTokenDataset(Dataset):
    """Concatenation of several TokenDatasets with optional per-file stride."""

    def __init__(
        self,
        paths: Sequence[PathLike],
        seq_len: int,
        dtype: str = "uint16",
        stride: int = 1,
    ):
        self.parts = [
            TokenDataset(p, seq_len=seq_len, dtype=dtype, stride=stride) for p in paths
        ]
        if not self.parts:
            raise ValueError("MultiFileTokenDataset: no paths")
        self._cum: list[int] = []
        total = 0
        for p in self.parts:
            total += len(p)
            self._cum.append(total)
        self.seq_len = seq_len
        self.n_tokens = sum(p.n_tokens for p in self.parts)

    def __len__(self) -> int:
        return self._cum[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        # binary search which file
        lo, hi = 0, len(self._cum)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        file_i = lo
        prev = 0 if file_i == 0 else self._cum[file_i - 1]
        return self.parts[file_i][idx - prev]


class StreamingTokenDataset(IterableDataset):
    """Infinite stream of random windows – useful for very large corpora."""

    def __init__(
        self,
        path: PathSpec,
        seq_len: int,
        dtype: str = "uint16",
        seed: int = 42,
    ):
        self.files = _resolve_token_files(path)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.seed = int(seed)
        self._arrays = [_open_mmap(f, _infer_dtype(f, dtype) if dtype == "auto" else dtype) for f in self.files]
        self._lengths = [max(0, len(a) - seq_len) for a in self._arrays]
        self.n_tokens = sum(len(a) for a in self._arrays)
        if sum(self._lengths) <= 0:
            raise ValueError(
                f"StreamingTokenDataset: all files too short for seq_len={seq_len}"
            )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        n_workers = info.num_workers if info else 1
        rank = torch.distributed.get_rank() if _dist_on() else 0
        rng = np.random.default_rng(self.seed + 1_000_003 * rank + 97 * worker)

        # weight sampling by usable length
        weights = np.array(self._lengths, dtype=np.float64)
        weights = weights / weights.sum()

        while True:
            fi = int(rng.choice(len(self._arrays), p=weights))
            hi = self._lengths[fi]
            if hi <= 0:
                continue
            # shard starts across workers roughly
            idx = int(rng.integers(0, hi))
            if n_workers > 1 and (idx % n_workers) != worker:
                # cheap re-roll rather than hard shard (keeps entropy high)
                idx = int(rng.integers(0, hi))
            chunk = self._arrays[fi][idx : idx + self.seq_len + 1]
            x = torch.from_numpy(np.asarray(chunk[:-1], dtype=np.int64))
            y = torch.from_numpy(np.asarray(chunk[1:], dtype=np.int64))
            yield {"input_ids": x, "targets": y}


def _dist_on() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def build_dataloader(
    path: Optional[PathSpec],
    seq_len: int,
    batch_size: int,
    streaming: bool = False,
    num_workers: int = 0,
    dtype: str = "uint16",
    shuffle_train: bool = True,
    stride: int = 1,
    prefetch_factor: Optional[int] = 2,
    drop_last: bool = True,
) -> Optional[DataLoader]:
    """Build a DataLoader over one or more token files / a directory.

    Args:
        path: file, directory of .bin/.npy, glob, or list of those.
        stride: window step for map-style datasets (1 = dense overlap).
        streaming: infinite random windows (better for multi-GB corpora).
    """
    if path is None:
        return None

    if streaming:
        rank = torch.distributed.get_rank() if _dist_on() else 0
        ds: Dataset | IterableDataset = StreamingTokenDataset(
            path, seq_len, dtype=dtype, seed=42 + 100_003 * rank
        )
        kwargs: dict = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = True
        ntok = getattr(ds, "n_tokens", 0)
        logger.info(
            "Streaming loader: ~%s tokens across %d file(s), seq_len=%d, batch=%d",
            f"{ntok:,}",
            len(getattr(ds, "files", []) or [1]),
            seq_len,
            batch_size,
        )
        return DataLoader(ds, **kwargs)

    files = _resolve_token_files(path)
    if len(files) == 1:
        ds = TokenDataset(files[0], seq_len=seq_len, dtype=dtype, stride=stride)
    else:
        ds = MultiFileTokenDataset(files, seq_len=seq_len, dtype=dtype, stride=stride)

    sampler = None
    if _dist_on() and shuffle_train:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=True, drop_last=drop_last)

    kwargs = {
        "batch_size": batch_size,
        "shuffle": sampler is None and shuffle_train,
        "sampler": sampler,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True

    ntok = getattr(ds, "n_tokens", 0)
    logger.info(
        "Map-style loader: %s samples, ~%s tokens, %d file(s), seq_len=%d, stride=%d, batch=%d",
        f"{len(ds):,}",
        f"{ntok:,}",
        len(files),
        seq_len,
        stride,
        batch_size,
    )
    return DataLoader(ds, **kwargs)


def estimate_tokens_per_step(batch_size: int, seq_len: int, world_size: int = 1) -> int:
    return int(batch_size) * int(seq_len) * max(1, int(world_size))


def count_tokens(path: PathSpec, dtype: str = "uint16") -> int:
    """Total tokens across one or more token files / a directory."""
    total = 0
    for f in _resolve_token_files(path):
        dt = _infer_dtype(f, dtype) if dtype == "auto" else dtype
        total += int(len(_open_mmap(f, dt)))
    return total

