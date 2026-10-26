"""Tensor Parallel (TP) and Pipeline Parallel (PP) helpers for SENTINEL.

Design
------
* **DDP (data parallel)** remains the default (see ``distributed.py``).
* **TP** shards selected Linear / Embedding weights across a TP group and
  all-reduces activations where needed (Megatron-style column / row parallel).
* **PP** splits one forward into ordered stages and runs **micro-batches**
  with P2P activation handoff (simple GPipe-style schedule).

World layout (optional)::

    WORLD_SIZE = data_parallel_size * tensor_parallel_size * pipeline_parallel_size

Launch example (2-way tensor parallel, 1 PP, rest DP)::

    torchrun --nproc_per_node=4 -m sentinel.cli train ... \\
      --tensor-parallel-size 2

Single-process runs are unchanged (tp=1, pp=1).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process groups
# ---------------------------------------------------------------------------


@dataclass
class ParallelState:
    """Ranks / groups for DP × TP × PP mesh."""

    rank: int = 0
    world_size: int = 1
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    dp_rank: int = 0
    tp_rank: int = 0
    pp_rank: int = 0
    tp_group: Any = None
    pp_group: Any = None
    dp_group: Any = None

    @property
    def is_tp(self) -> bool:
        return self.tp_size > 1 and self.tp_group is not None

    @property
    def is_pp(self) -> bool:
        return self.pp_size > 1 and self.pp_group is not None


_STATE = ParallelState()


def get_parallel_state() -> ParallelState:
    return _STATE


def _world_rank() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def init_parallel_groups(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> ParallelState:
    """Build TP / PP / DP groups. Call after ``init_distributed_if_needed``."""
    global _STATE
    rank, world = _world_rank()
    tp = max(1, int(tensor_parallel_size))
    pp = max(1, int(pipeline_parallel_size))
    if world % (tp * pp) != 0:
        raise ValueError(
            f"WORLD_SIZE={world} must be divisible by tp*pp={tp * pp} "
            f"(tensor_parallel_size={tp}, pipeline_parallel_size={pp})"
        )
    dp = world // (tp * pp)

    # Rank layout: global = dp_rank * (tp*pp) + pp_rank * tp + tp_rank
    tp_rank = rank % tp
    pp_rank = (rank // tp) % pp
    dp_rank = rank // (tp * pp)

    tp_group = pp_group = dp_group = None
    if dist.is_available() and dist.is_initialized() and world > 1:
        # TP groups: same (dp, pp), vary tp
        for d in range(dp):
            for p in range(pp):
                ranks = [d * (tp * pp) + p * tp + t for t in range(tp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    tp_group = g
        # PP groups: same (dp, tp), vary pp
        for d in range(dp):
            for t in range(tp):
                ranks = [d * (tp * pp) + p * tp + t for p in range(pp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    pp_group = g
        # DP groups: same (tp, pp), vary dp
        for p in range(pp):
            for t in range(tp):
                ranks = [d * (tp * pp) + p * tp + t for d in range(dp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    dp_group = g

    _STATE = ParallelState(
        rank=rank,
        world_size=world,
        dp_size=dp,
        tp_size=tp,
        pp_size=pp,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        tp_group=tp_group,
        pp_group=pp_group,
        dp_group=dp_group,
    )
    if rank == 0:
        logger.info(
            "parallel mesh: world=%d dp=%d tp=%d pp=%d",
            world, dp, tp, pp,
        )
    return _STATE


# ---------------------------------------------------------------------------
# Tensor parallel layers
# ---------------------------------------------------------------------------


def _tp_all_reduce(x: Tensor, group: Any) -> Tensor:
    if group is None or not dist.is_initialized():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


def _tp_split_dim0(weight: Tensor, tp_rank: int, tp_size: int) -> Tensor:
    assert weight.size(0) % tp_size == 0, f"out_features {weight.size(0)} not divisible by tp={tp_size}"
    chunk = weight.size(0) // tp_size
    return weight[tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def _tp_split_dim1(weight: Tensor, tp_rank: int, tp_size: int) -> Tensor:
    assert weight.size(1) % tp_size == 0, f"in_features {weight.size(1)} not divisible by tp={tp_size}"
    chunk = weight.size(1) // tp_size
    return weight[:, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


class ColumnParallelLinear(nn.Module):
    """Y = X @ W^T with W split on **output** dim (column parallel)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        gather_output: bool = True,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: Any = None,
    ):
        super().__init__()
        assert out_features % max(1, tp_size) == 0
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_rank = tp_rank
        self.tp_size = max(1, tp_size)
        self.tp_group = tp_group
        local_out = out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(local_out, in_features))
        self.bias = nn.Parameter(torch.empty(local_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            parts = [torch.empty_like(y) for _ in range(self.tp_size)]
            dist.all_gather(parts, y.contiguous(), group=self.tp_group)
            # order by tp rank
            y = torch.cat(parts, dim=-1)
        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        gather_output: bool = True,
        state: Optional[ParallelState] = None,
    ) -> "ColumnParallelLinear":
        st = state or get_parallel_state()
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            gather_output=gather_output,
            tp_rank=st.tp_rank,
            tp_size=st.tp_size,
            tp_group=st.tp_group,
        )
        with torch.no_grad():
            w = _tp_split_dim0(linear.weight.data, st.tp_rank, st.tp_size)
            mod.weight.copy_(w)
            if linear.bias is not None and mod.bias is not None:
                b = linear.bias.data
                chunk = b.size(0) // st.tp_size
                mod.bias.copy_(b[st.tp_rank * chunk : (st.tp_rank + 1) * chunk])
        return mod


class RowParallelLinear(nn.Module):
    """Y = X @ W^T with W split on **input** dim; output all-reduced."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        input_is_parallel: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: Any = None,
    ):
        super().__init__()
        assert in_features % max(1, tp_size) == 0
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tp_rank = tp_rank
        self.tp_size = max(1, tp_size)
        self.tp_group = tp_group
        local_in = in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(out_features, local_in))
        # bias only on rank 0 to avoid double-add after all-reduce
        self.bias = nn.Parameter(torch.empty(out_features)) if bias and tp_rank == 0 else None
        self._bias_out = out_features if bias else 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if not self.input_is_parallel and self.tp_size > 1:
            chunk = x.size(-1) // self.tp_size
            x = x[..., self.tp_rank * chunk : (self.tp_rank + 1) * chunk].contiguous()
        y = F.linear(x, self.weight, None)
        if self.tp_size > 1 and self.tp_group is not None:
            y = _tp_all_reduce(y, self.tp_group)
        if self.bias is not None:
            y = y + self.bias
        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        input_is_parallel: bool = False,
        state: Optional[ParallelState] = None,
    ) -> "RowParallelLinear":
        st = state or get_parallel_state()
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            input_is_parallel=input_is_parallel,
            tp_rank=st.tp_rank,
            tp_size=st.tp_size,
            tp_group=st.tp_group,
        )
        with torch.no_grad():
            w = _tp_split_dim1(linear.weight.data, st.tp_rank, st.tp_size)
            mod.weight.copy_(w)
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(linear.bias.data)
        return mod


class VocabParallelEmbedding(nn.Module):
    """Embedding sharded on vocab dimension."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: Any = None,
    ):
        super().__init__()
        assert num_embeddings % max(1, tp_size) == 0
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = tp_rank
        self.tp_size = max(1, tp_size)
        self.tp_group = tp_group
        self.local_vocab = num_embeddings // self.tp_size
        self.vocab_start = self.tp_rank * self.local_vocab
        self.vocab_end = self.vocab_start + self.local_vocab
        self.weight = nn.Parameter(torch.empty(self.local_vocab, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        if self.tp_size == 1:
            return F.embedding(input_ids, self.weight)
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).clamp(min=0)
        out = F.embedding(local_ids, self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        if self.tp_group is not None:
            out = _tp_all_reduce(out, self.tp_group)
        return out

    @classmethod
    def from_embedding(
        cls, emb: nn.Embedding, state: Optional[ParallelState] = None
    ) -> "VocabParallelEmbedding":
        st = state or get_parallel_state()
        mod = cls(
            emb.num_embeddings,
            emb.embedding_dim,
            tp_rank=st.tp_rank,
            tp_size=st.tp_size,
            tp_group=st.tp_group,
        )
        with torch.no_grad():
            w = emb.weight.data
            mod.weight.copy_(w[mod.vocab_start : mod.vocab_end])
        return mod


def apply_tensor_parallel(model: nn.Module, state: Optional[ParallelState] = None) -> nn.Module:
    """Shard SENTINEL output head (+ tied embed) for TP. Safe no-op if tp_size==1."""
    st = state or get_parallel_state()
    if st.tp_size <= 1:
        return model

    # Output head: column-parallel then gather (logits full vocab on each rank for CE)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        if model.head.out_features % st.tp_size != 0:
            logger.warning(
                "head.out_features=%d not divisible by tp=%d; skip head TP",
                model.head.out_features,
                st.tp_size,
            )
        else:
            model.head = ColumnParallelLinear.from_linear(
                model.head, gather_output=True, state=st
            )
            logger.info("TP: sharded lm head (column parallel, gather)")

    # Skill MLP first linear (expand) column-parallel; second row-parallel
    router = getattr(model, "router", None)
    skills = getattr(router, "skills", None) if router is not None else None
    if skills is not None:
        for i, sk in enumerate(skills):
            net = getattr(sk, "net", None)
            if net is None or len(net) < 3:
                continue
            if isinstance(net[0], nn.Linear) and net[0].out_features % st.tp_size == 0:
                if isinstance(net[2], nn.Linear) and net[2].in_features % st.tp_size == 0:
                    col = ColumnParallelLinear.from_linear(
                        net[0], gather_output=False, state=st
                    )
                    row = RowParallelLinear.from_linear(
                        net[2], input_is_parallel=True, state=st
                    )
                    sk.net = nn.Sequential(col, net[1], row)
        logger.info("TP: sharded skill MLPs (column+row)")

    return model


# ---------------------------------------------------------------------------
# Pipeline parallel (micro-batch GPipe-style)
# ---------------------------------------------------------------------------


def split_microbatches(batch: Tensor, n_micro: int) -> List[Tensor]:
    """Split batch dim into n_micro chunks (last may be smaller)."""
    n_micro = max(1, int(n_micro))
    if n_micro == 1 or batch.size(0) < n_micro:
        return [batch]
    return list(torch.chunk(batch, n_micro, dim=0))


def pipeline_p2p_send(tensor: Tensor, dst: int, group: Any) -> None:
    if group is None or not dist.is_initialized():
        return
    dist.send(tensor.contiguous(), dst=dst, group=group)


def pipeline_p2p_recv(
    template: Tensor, src: int, group: Any
) -> Tensor:
    if group is None or not dist.is_initialized():
        return template
    out = torch.empty_like(template)
    dist.recv(out, src=src, group=group)
    return out


def pipeline_rank_global(state: ParallelState, pp_rank: int) -> int:
    """Global rank for a peer at the same (dp, tp) with given pp_rank."""
    return state.dp_rank * (state.tp_size * state.pp_size) + pp_rank * state.tp_size + state.tp_rank


def run_pipeline_stages(
    stages: Sequence[Callable[[Tensor], Tensor]],
    x: Tensor,
    *,
    n_micro: int = 1,
    state: Optional[ParallelState] = None,
) -> Tensor:
    """GPipe-style: each PP rank runs its stage on micro-batches.

    * ``stages[i]`` must be callable on the activation for pipeline stage i.
    * Only the local stage ``stages[pp_rank]`` is invoked on this rank.
    * Activations are sent to the next stage rank.

    For single-process (pp_size==1) runs all stages in order on ``x``.
    """
    st = state or get_parallel_state()
    if st.pp_size <= 1 or len(stages) == 1:
        y = x
        for fn in stages:
            y = fn(y)
        return y

    assert len(stages) >= st.pp_size
    micros = split_microbatches(x, n_micro)
    outs: List[Tensor] = []
    local_fn = stages[st.pp_rank]
    prev = st.pp_rank - 1
    nxt = st.pp_rank + 1

    for mb in micros:
        act = mb
        if st.pp_rank > 0:
            # receive activation shaped like local stage expects — use mb as shape hint
            # Caller should pass float activations; for input_ids (long) stage0 only
            act = pipeline_p2p_recv(mb, src=pipeline_rank_global(st, prev), group=st.pp_group)
        act = local_fn(act)
        if st.pp_rank < st.pp_size - 1:
            pipeline_p2p_send(act, dst=pipeline_rank_global(st, nxt), group=st.pp_group)
        else:
            outs.append(act)

    if st.pp_rank == st.pp_size - 1:
        return torch.cat(outs, dim=0)
    # Non-last stages return empty placeholder (loss computed on last stage)
    return x.new_zeros(0)
