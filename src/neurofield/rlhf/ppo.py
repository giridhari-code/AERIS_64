"""PPO-style policy loss with KL to reference (simplified, single-step)."""

from __future__ import annotations

import torch
from torch import Tensor


def ppo_policy_loss(
    logprob_new: Tensor,
    logprob_old: Tensor,
    advantage: Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
    logprob_ref: Tensor | None = None,
) -> Tensor:
    """
    logprob_* : scalars or (,) sums over tokens for one sample
    advantage : scalar
    """
    ratio = torch.exp(logprob_new - logprob_old.detach())
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy = -torch.min(unclipped, clipped)
    kl = torch.tensor(0.0, device=policy.device)
    if logprob_ref is not None:
        # approx KL(policy || ref) ~ logpi - logref
        kl = kl_coef * (logprob_new - logprob_ref.detach())
    return policy + kl
