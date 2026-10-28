"""ModelRunner — prefill/decode execution over SENTINEL (DeepInfer worker/model_runner role)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import torch

from sentinel.model import SENTINEL
from sentinel.model_executor.output import ModelStepOutput

TokenIds = Union[List[int], Sequence[int]]


@dataclass
class RunnerConfig:
    write_slow: bool = True
    collect_stats: bool = False


class ModelRunner:
    """Owns device forward only. Sampling stays in sampling.py / engine."""

    def __init__(
        self,
        model: SENTINEL,
        device: torch.device,
        amp_dtype: Optional[torch.dtype] = None,
        config: Optional[RunnerConfig] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.amp_dtype = amp_dtype
        self.config = config or RunnerConfig()
        self.model.eval()

    def _autocast(self):
        use = self.amp_dtype is not None and self.device.type == "cuda"
        if use:
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return nullcontext()

    def _forward(self, token_ids: TokenIds, state: Any) -> ModelStepOutput:
        ids = [int(x) for x in token_ids]
        if not ids:
            raise ValueError("empty token_ids")
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            with self._autocast():
                out = self.model(
                    x,
                    state=state,
                    targets=None,
                    collect_stats=self.config.collect_stats,
                    write_slow=self.config.write_slow,
                )
        logits = out.logits[0, -1].float()
        return ModelStepOutput(
            logits=logits,
            state=out.state,
            tokens_processed=len(ids),
        )

    def prefill(self, prompt_ids: TokenIds) -> ModelStepOutput:
        """Full prompt — state starts None."""
        return self._forward(prompt_ids, state=None)

    def decode(self, token_id: int, state: Any) -> ModelStepOutput:
        """Single-token step with carried state."""
        return self._forward([int(token_id)], state=state)

    def step(self, token_ids: TokenIds, state: Any = None) -> ModelStepOutput:
        """Generic step: prefill if state is None and len>1, else decode-style."""
        return self._forward(token_ids, state=state)
