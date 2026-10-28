"""Executor output types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class ModelStepOutput:
    """One prefill or decode step."""

    logits: torch.Tensor  # (V,) float32 last-position logits
    state: Any  # SENTINEL recurrent state dict
    tokens_processed: int = 0
