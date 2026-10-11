"""GPU / accelerator setup for production training.

Call ``configure_accelerator`` once before building the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AcceleratorConfig:
    device: torch.device
    use_bf16: bool
    use_compile: bool
    tf32: bool


def configure_accelerator(
    device_str: str = "auto",
    *,
    bf16: Optional[bool] = None,
    compile_model: bool = False,
    tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
) -> AcceleratorConfig:
    """
    device_str: "cpu" | "cuda" | "cuda:0" | "auto"
    bf16: None = on when CUDA and bf16 supported; False forces off
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

    if device.type == "cuda":
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = bool(tf32)
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass
        # Avoid fragmentation on long Colab runs
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if bf16 is None:
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    else:
        use_bf16 = bool(bf16) and (
            device.type == "cpu"
            or (device.type == "cuda" and torch.cuda.is_bf16_supported())
        )

    use_compile = bool(compile_model) and hasattr(torch, "compile")
    return AcceleratorConfig(
        device=device, use_bf16=use_bf16, use_compile=use_compile, tf32=tf32
    )


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    try:
        return torch.compile(model, mode="default")
    except Exception as e:
        print(f"[gpu] torch.compile skipped: {e}")
        return model


def autocast_ctx(cfg: AcceleratorConfig):
    if not cfg.use_bf16:
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16)
