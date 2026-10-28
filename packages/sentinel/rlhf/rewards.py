"""Reward / preference helpers: masked sequence log-probs, log-odds and pairwise ranking loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def token_logprobs(model, input_ids: Tensor, target_ids: Tensor) -> Tensor:
    """(B, T) teacher-forced log p(target_t | input_<=t). No LM loss is computed."""
    out = model(input_ids, lambda_err=0.0, lambda_bal=0.0, collect_stats=False)
    logits = out.logits
    b, t, v = logits.shape
    nll = F.cross_entropy(logits.float().reshape(-1, v), target_ids.reshape(-1), reduction="none")
    return -nll.view(b, t)


def reduce_logprobs(logp: Tensor, mask: Tensor | None = None, reduce: str = "sum") -> Tensor:
    """Reduce (B, T) token log-probs to (B,) over the masked (response) positions only."""
    m = torch.ones_like(logp) if mask is None else mask.to(device=logp.device, dtype=logp.dtype)
    total = (logp * m).sum(dim=-1)
    if reduce == "sum":
        return total
    if reduce == "mean":
        return total / m.sum(dim=-1).clamp(min=1.0)
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


def sequence_logprob(
    model,
    input_ids: Tensor,
    target_ids: Tensor,
    mask: Tensor | None = None,
    reduce: str = "mean",
    eval_mode: bool = False,
) -> Tensor:
    """(B,) log-prob of target_ids, over `mask` positions (response tokens) if given.

    mask=None keeps the old behaviour (every position, padding included) - pass the mask from
    preference_loader. eval_mode=True scores with dropout / train-only gating off (gradients still flow),
    so policy and reference are scored in the same mode.
    """
    was_training = bool(getattr(model, "training", False))
    switch = eval_mode and was_training
    if switch:
        model.eval()
    try:
        logp = token_logprobs(model, input_ids, target_ids)
    finally:
        if switch:
            model.train()
    return reduce_logprobs(logp, mask, reduce)


def log_odds(logp: Tensor) -> Tensor:
    """log(p / (1 - p)) with p = exp(logp); logp is a length-normalised (mean) log-prob, i.e. < 0."""
    lp = logp.clamp(max=-1e-6)
    return lp - torch.log1p(-torch.exp(lp))


def pairwise_reward_loss(
    model,
    chosen_x: Tensor,
    chosen_y: Tensor,
    rejected_x: Tensor,
    rejected_y: Tensor,
    chosen_m: Tensor | None = None,
    rejected_m: Tensor | None = None,
) -> Tensor:
    """Bradley-Terry: prefer higher (mean, response-only) logprob on chosen than rejected."""
    s_c = sequence_logprob(model, chosen_x, chosen_y, chosen_m, "mean", eval_mode=True)
    s_r = sequence_logprob(model, rejected_x, rejected_y, rejected_m, "mean", eval_mode=True)
    return -F.logsigmoid(s_c - s_r).mean()
