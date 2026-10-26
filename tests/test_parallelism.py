"""Unit tests for TP/PP helpers (single-process; no multi-GPU required)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sentinel.training.parallelism import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VocabParallelEmbedding,
    apply_tensor_parallel,
    get_parallel_state,
    init_parallel_groups,
    split_microbatches,
)


def test_init_parallel_default_mesh():
    st = init_parallel_groups(1, 1)
    assert st.tp_size == 1 and st.pp_size == 1 and st.dp_size >= 1


def test_split_microbatches():
    x = torch.arange(8).view(4, 2)
    parts = split_microbatches(x, 2)
    assert len(parts) == 2
    assert parts[0].shape[0] == 2


def test_column_parallel_from_linear_tp1():
    lin = nn.Linear(16, 32)
    st = ParallelState(tp_size=1, tp_rank=0)
    col = ColumnParallelLinear.from_linear(lin, state=st)
    x = torch.randn(3, 16)
    y = col(x)
    assert y.shape == (3, 32)
    with torch.no_grad():
        ref = lin(x)
    assert torch.allclose(y, ref, atol=1e-5)


def test_row_parallel_from_linear_tp1():
    lin = nn.Linear(16, 8)
    st = ParallelState(tp_size=1, tp_rank=0)
    row = RowParallelLinear.from_linear(lin, state=st)
    x = torch.randn(3, 16)
    y = row(x)
    assert y.shape == (3, 8)


def test_apply_tensor_parallel_noop_tp1():
    from sentinel.config import ModelConfig
    from sentinel.model import SENTINEL

    m = SENTINEL(ModelConfig(
        d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1,
        k_max=1, dendrite_window=2, vocab_size=64, max_seq_len=32,
    ))
    st = init_parallel_groups(1, 1)
    apply_tensor_parallel(m, st)
    x = torch.randint(0, 64, (1, 8))
    out = m(x)
    assert out.logits.shape[-1] == 64


def test_vocab_parallel_embedding_tp1():
    emb = nn.Embedding(100, 16)
    st = ParallelState(tp_size=1, tp_rank=0)
    vp = VocabParallelEmbedding.from_embedding(emb, st)
    ids = torch.randint(0, 100, (2, 5))
    assert vp(ids).shape == (2, 5, 16)
