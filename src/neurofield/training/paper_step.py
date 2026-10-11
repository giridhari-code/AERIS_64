"""One training step with paper v3 slow-memory replay."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def forward_train(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    lambda_err: float = 0.03,
    lambda_bal: float = 0.005,
) -> Any:
    return model(x, targets=y, lambda_err=lambda_err, lambda_bal=lambda_bal)


def maybe_replay(
    model: torch.nn.Module,
    out: Any,
    step: int,
    *,
    replay_every: int = 50,
    eta: float = 0.02,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    """Appendix A: W_s ← (1-η)W_s + η weighted M."""
    if step % max(1, replay_every) != 0:
        return
    if out is None or out.state is None or "M" not in out.state:
        return
    S = (out.metrics or {}).get("_seq_surprise")
    if S is None or not torch.is_tensor(S):
        b = batch_size or out.state["M"].size(0)
        dev = device or out.state["M"].device
        S = torch.ones(b, device=dev)
    model.replay_slow(out.state["M"], S, eta)
