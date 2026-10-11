from neurofield.training.gpu import configure_accelerator, maybe_compile, autocast_ctx
import torch
import torch.nn as nn

def test_configure_cpu():
    a = configure_accelerator("cpu", bf16=False, compile_model=False)
    assert a.device.type == "cpu"
    assert a.use_bf16 is False

def test_maybe_compile_noop():
    m = nn.Linear(4, 4)
    m2 = maybe_compile(m, False)
    assert m2 is m

def test_autocast_ctx_runs():
    a = configure_accelerator("cpu", bf16=False)
    with autocast_ctx(a):
        x = torch.randn(2, 4)
        assert x.shape == (2, 4)
