"""Wiring tests for model_executor + serve lifecycle hooks."""

from __future__ import annotations

import torch

from sentinel.config import ModelConfig
from sentinel.model import SENTINEL
from sentinel.model_executor import (
    ModelRunner,
    SamplingParams,
    sample_from_logits,
)
import sentinel.serving.server as srv


def _tiny():
    cfg = ModelConfig(
        d_model=32,
        d_k=16,
        d_v=16,
        n_skills=2,
        top_k=1,
        k_max=1,
        dendrite_window=2,
        vocab_size=64,
        max_seq_len=64,
    )
    return SENTINEL(cfg).eval()


def test_runner_prefill_decode():
    m = _tiny()
    r = ModelRunner(m, torch.device("cpu"))
    o = r.prefill([1, 2, 3, 4])
    assert o.logits.shape == (64,)
    assert o.state is not None
    o2 = r.decode(5, o.state)
    assert o2.logits.shape == (64,)
    assert o2.tokens_processed == 1


def test_sample_greedy_deterministic():
    logits = torch.zeros(16)
    logits[7] = 10.0
    tid = sample_from_logits(logits, SamplingParams(temperature=0.0))
    assert tid == 7


def test_engine_lifecycle_wired():
    m = _tiny()
    srv._runner = ModelRunner(m, torch.device("cpu"))
    srv._encode = lambda t: [min(63, (ord(c) % 64)) for c in (t or "x")][:32] or [1]
    srv._decode = lambda ids: "".join(chr(32 + (i % 64)) for i in ids)
    srv._device = torch.device("cpu")
    srv._model_id = "t"
    srv._stop_ids = set()
    seq = srv.add_request("hi", SamplingParams(max_tokens=4, temperature=0.0))
    out = srv.engine_run(seq)
    assert out.status == srv.SequenceStatus.FINISHED
    assert len(out.output_ids) == 4
    assert out.finish_reason in ("stop", "length")
