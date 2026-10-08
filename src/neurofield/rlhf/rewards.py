"""Reward model helpers: sequence log-prob and pairwise ranking loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprob(model, input_ids: Tensor, target_ids: Tensor) -> Tensor:
    """Mean token log-prob of target_ids given input_ids prefix batch.

    input_ids: (B, T_in)  — prompt
    We score teacher-forced on concat prompt+response; returns (B,) mean logprob on response span.
    """
    # For simplicity: teacher-force on full sequence and return mean logprob per sequence
    # input_ids here is full sequence (prompt+response)[:, :-1], targets [:, 1:]
    out = model(input_ids, targets=target_ids, lambda_err=0.0, lambda_bal=0.0)
    # reconstruct token logprobs from logits
    logits = out.logits  # (B, T, V)
    logp = F.log_softmax(logits, dim=-1)
    gather = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
    # mask pad zeros if any — mean over time
    return gather.mean(dim=-1)


def pairwise_reward_loss(
    model,
    chosen_x: Tensor,
    chosen_y: Tensor,
    rejected_x: Tensor,
    rejected_y: Tensor,
) -> Tensor:
    """Bradley-Terry: prefer higher logprob on chosen than rejected."""
    s_c = sequence_logprob(model, chosen_x, chosen_y)
    s_r = sequence_logprob(model, rejected_x, rejected_y)
    # -log sigmoid(s_c - s_r)
    return -F.logsigmoid(s_c - s_r).mean()
