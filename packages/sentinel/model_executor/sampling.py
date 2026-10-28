"""Sampling utilities (optimized paths for greedy / top_p=1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    seed: Optional[int] = None


def sample_from_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: Optional[torch.Generator] = None,
) -> int:
    """logits: (V,) on CPU or CUDA."""
    temperature = float(params.temperature)
    # Fast greedy
    if temperature <= 1e-6:
        return int(torch.argmax(logits).item())

    logits = logits / max(temperature, 1e-5)
    k = int(params.top_k)
    v = logits.numel()
    if 0 < k < v:
        vals, idx = torch.topk(logits, k)
        masked = torch.full_like(logits, float("-inf"))
        masked[idx] = vals
        logits = masked

    p = float(params.top_p)
    if 0.0 < p < 1.0:
        # Only sort when nucleus is active
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum > p
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    if generator is not None:
        return int(torch.multinomial(probs, 1, generator=generator).item())
    return int(torch.multinomial(probs, 1).item())
