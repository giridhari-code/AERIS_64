"""Direct Preference Optimization (Rafailov et al.)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from sentinel.rlhf.rewards import sequence_logprob


def dpo_loss(
    policy,
    ref,
    chosen_x: Tensor,
    chosen_y: Tensor,
    rejected_x: Tensor,
    rejected_y: Tensor,
    beta: float = 0.1,
    chosen_m: Tensor | None = None,
    rejected_m: Tensor | None = None,
) -> Tensor:
    """
    L = -log σ( β * ( log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ) )

    Log-probs are SUMS over the response tokens (masks from preference_loader), as in the paper;
    policy and reference are both scored without dropout, so the loss starts at log 2.
    """
    with torch.no_grad():
        ref.train(False)
        r_c = sequence_logprob(ref, chosen_x, chosen_y, chosen_m, "sum")
        r_r = sequence_logprob(ref, rejected_x, rejected_y, rejected_m, "sum")
    p_c = sequence_logprob(policy, chosen_x, chosen_y, chosen_m, "sum", eval_mode=True)
    p_r = sequence_logprob(policy, rejected_x, rejected_y, rejected_m, "sum", eval_mode=True)
    pi_c = p_c - r_c
    pi_r = p_r - r_r
    return -F.logsigmoid(beta * (pi_c - pi_r)).mean()
