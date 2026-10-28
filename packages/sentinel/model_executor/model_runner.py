"""ModelRunner — optimized prefill/decode over SENTINEL."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

import torch

from sentinel.model import SENTINEL
from sentinel.model_executor.output import ModelStepOutput

TokenIds = Union[List[int], Sequence[int]]


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RunnerConfig:
    write_slow: bool = True
    collect_stats: bool = False
    # NEUROFIELD_TORCH_COMPILE=1 enables optional torch.compile (CUDA; first call slow)
    torch_compile: bool = False


class ModelRunner:
    """Device execution. Sampling stays outside unless generate() is used."""

    def __init__(
        self,
        model: SENTINEL,
        device: torch.device,
        amp_dtype: Optional[torch.dtype] = None,
        config: Optional[RunnerConfig] = None,
    ) -> None:
        self.device = device
        self.amp_dtype = amp_dtype
        self.config = config or RunnerConfig(
            torch_compile=_env_truthy("NEUROFIELD_TORCH_COMPILE", "0"),
        )
        model.eval()
        # Optional compile — skip on CPU by default (often slower for tiny models)
        if self.config.torch_compile and device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
            except Exception:
                pass
        self.model = model
        # Reused 1-token buffer for decode (avoids alloc every step)
        self._decode_buf = torch.zeros(1, 1, dtype=torch.long, device=device)

    def _autocast(self):
        if self.amp_dtype is not None and self.device.type == "cuda":
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return nullcontext()

    def _forward(self, token_ids: TokenIds, state: Any) -> ModelStepOutput:
        ids = list(token_ids)
        if not ids:
            raise ValueError("empty token_ids")
        n = len(ids)
        if n == 1:
            self._decode_buf[0, 0] = int(ids[0])
            x = self._decode_buf
        else:
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
        # Keep logits on device; caller samples there when possible
        logits = out.logits[0, -1]
        if logits.dtype != torch.float32:
            logits = logits.float()
        return ModelStepOutput(
            logits=logits,
            state=out.state,
            tokens_processed=n,
        )

    def prefill(self, prompt_ids: TokenIds) -> ModelStepOutput:
        return self._forward(prompt_ids, state=None)

    def decode(self, token_id: int, state: Any) -> ModelStepOutput:
        self._decode_buf[0, 0] = int(token_id)
        with torch.inference_mode():
            with self._autocast():
                out = self.model(
                    self._decode_buf,
                    state=state,
                    targets=None,
                    collect_stats=False,
                    write_slow=self.config.write_slow,
                )
        logits = out.logits[0, -1]
        if logits.dtype != torch.float32:
            logits = logits.float()
        return ModelStepOutput(logits=logits, state=out.state, tokens_processed=1)

    def step(self, token_ids: TokenIds, state: Any = None) -> ModelStepOutput:
        return self._forward(token_ids, state=state)

    def generate(
        self,
        prompt_ids: TokenIds,
        max_new: int,
        sample_fn: Callable[[torch.Tensor], int],
        stop_ids: Optional[set[int]] = None,
    ) -> tuple[List[int], Any, str]:
        """
        Tight prefill+decode loop under one inference_mode region.

        sample_fn(logits_v) -> token id
        Returns (output_ids, final_state, finish_reason).
        """
        stop_ids = stop_ids or set()
        max_new = max(1, min(int(max_new), 256))
        out_ids: List[int] = []
        with torch.inference_mode():
            step = self.prefill(prompt_ids)
            state = step.state
            tid = int(sample_fn(step.logits))
            if tid in stop_ids:
                return out_ids, state, "stop"
            out_ids.append(tid)
            while len(out_ids) < max_new:
                step = self.decode(out_ids[-1], state)
                state = step.state
                tid = int(sample_fn(step.logits))
                if tid in stop_ids:
                    return out_ids, state, "stop"
                out_ids.append(tid)
        return out_ids, state, "length"
