"""Training-parallelism tests: token-parallel forward path + data-parallel (DDP, gloo/CPU)."""

from __future__ import annotations

import os
import socket

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from neurofield.config import ModelConfig, NeuroFieldConfig, SafetyConfig  # noqa: E402
from neurofield.model import NeuroField  # noqa: E402


def _cfg(**kw):
    return ModelConfig(d_model=48, d_k=16, d_v=16, n_skills=4, top_k=2, k_max=3, vocab_size=300,
                       dendrite_window=4, max_seq_len=64, dropout=0.0, **kw)


def test_parallel_ops_is_numerically_identical_to_the_per_token_path():
    torch.manual_seed(0)
    m = NeuroField(_cfg(), SafetyConfig(enable_audit=False))
    m.meta.track_global = False
    x, y = torch.randint(0, 300, (3, 40)), torch.randint(0, 300, (3, 40))
    res = {}
    for flag in (False, True):
        m.parallel_ops = flag
        m.zero_grad(set_to_none=True)
        out = m(x, targets=y)
        out.loss.backward()
        res[flag] = (out, torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    a, b = res[False], res[True]
    assert torch.allclose(a[0].logits, b[0].logits, atol=2e-5, rtol=1e-5)
    assert torch.allclose(a[0].loss, b[0].loss, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a[1], b[1], atol=1e-4)  # gradients: identical up to float summation order
    assert torch.allclose(a[0].state["x_hist"], b[0].state["x_hist"], atol=1e-6)
    assert torch.equal(a[0].state["slow_filled"], b[0].state["slow_filled"])


def test_parallel_ops_chunked_state_continuation_matches_per_token_path():
    """Carrying state across calls (prefill + decode style) gives the same result in both paths.

    (Chunked != one long call even in the legacy path, by design: a multi-token call snapshots the slow ring
    at its last token. What must hold is parallel == per-token for the SAME chunking.)
    """
    torch.manual_seed(1)
    m = NeuroField(_cfg(), SafetyConfig(enable_audit=False)).eval()
    x = torch.randint(0, 300, (2, 30))
    outs = {}
    with torch.no_grad():
        for flag in (False, True):
            m.parallel_ops = flag
            first = m(x[:, :11])
            second = m(x[:, 11:], state=first.state)
            third = m(x[:, :1], state=second.state)  # a 1-token decode step on top
            outs[flag] = (first.logits, second.logits, third.logits)
    for a, b in zip(outs[False], outs[True]):
        assert torch.allclose(a, b, atol=1e-5)




def test_read_seq_matches_per_token_read():
    torch.manual_seed(2)
    m = NeuroField(_cfg(), SafetyConfig(enable_audit=False)).eval()
    sm = m.slow_mem
    B, T = 3, 9
    D = torch.randn(B, T, m.cfg.d_model)
    slots = torch.randn(B, sm.n_slots, m.cfg.d_k, m.cfg.d_v)
    filled = torch.zeros(B, sm.n_slots)
    filled[:, :2] = 1.0
    filled[2] = 0.0
    sm.M_s.data.normal_()
    with torch.no_grad():
        ref = torch.stack([sm.read(D[:, t], slots=slots, filled=filled) for t in range(T)], dim=1)
        got = sm.read_seq(D, slots=slots, filled=filled)
        ref0 = torch.stack([sm.read(D[:, t]) for t in range(T)], dim=1)
        got0 = sm.read_seq(D)
    assert torch.allclose(ref, got, atol=1e-5)
    assert torch.allclose(ref0, got0, atol=1e-5)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, out_dir: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK="0")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from neurofield.training import Trainer

    torch.manual_seed(0)  # identical init on every rank (DDP also broadcasts rank 0)
    cfg = NeuroFieldConfig()
    t = cfg.training
    t.device, t.max_steps, t.warmup_steps, t.replay_every = "cpu", 6, 1, 3
    t.log_every, t.eval_every, t.checkpoint_every, t.grad_accum_steps = 100, 1000, 1000, 2
    t.lambda_err_warmup_steps, t.output_dir = 4, out_dir
    model = NeuroField(
        ModelConfig(d_model=24, d_k=8, d_v=8, n_skills=2, top_k=1, k_max=2, vocab_size=20,
                    max_seq_len=16, dendrite_window=2), cfg.safety)
    g = torch.Generator().manual_seed(100 + rank)  # DIFFERENT data on each rank
    loader = [{"input_ids": torch.randint(0, 20, (2, 12), generator=g),
               "targets": torch.randint(0, 20, (2, 12), generator=g)} for _ in range(30)]
    Trainer(model, cfg, loader).train()

    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [None] * world
    dist.all_gather_object(gathered, flat)
    assert all(torch.allclose(gathered[0], g_, atol=1e-6) for g_ in gathered), "ranks diverged"
    assert float(model.slow_mem.M_s.abs().sum()) > 0, "replay never ran"
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo unavailable")
def test_ddp_two_ranks_stay_identical_with_accumulation_replay_and_warmup(tmp_path):
    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / "final").exists()  # rank 0 exported the model


def test_lambda_err_warmup_field_defaults_off():
    assert NeuroFieldConfig().training.lambda_err_warmup_steps == 0
