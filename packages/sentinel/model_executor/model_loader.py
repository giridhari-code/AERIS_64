"""Load SENTINEL weights + tokenizer (DeepInfer model_loader role)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from sentinel.checkpoint.resolve import load_model_from_checkpoint
from sentinel.model import SENTINEL

logger = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(f"NEUROFIELD_{key}") or os.environ.get(key) or default).strip()


@dataclass
class LoadedModel:
    model: SENTINEL
    encode: Callable[..., Any]
    decode: Callable[..., Any]
    device: torch.device
    amp_dtype: Optional[torch.dtype]
    model_id: str
    checkpoint: str
    meta: dict


def load_model(
    checkpoint: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
    *,
    strict: Optional[bool] = None,
) -> LoadedModel:
    """Load from safetensors folder (or supported resolve path)."""
    want = device or "cpu"
    dev = torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")
    if strict is None:
        strict = _env("STRICT_LOAD", "1") == "1"
    model, tok, _cfg, meta = load_model_from_checkpoint(
        checkpoint,
        config_path,
        device=str(dev),
        strict=strict,
    )
    model.eval()
    dtype_name = _env("DTYPE", "float32")
    amp_map = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in amp_map:
        raise ValueError(f"Unknown NEUROFIELD_DTYPE={dtype_name}")
    if hasattr(model, "meta") and model.meta is not None:
        try:
            model.meta.track_global = False
        except Exception:
            pass
    model_id = str(meta.get("model_id") or meta.get("name") or Path(checkpoint).stem or "model")
    logger.info("model_executor loaded %s from %s on %s", model_id, checkpoint, dev)
    return LoadedModel(
        model=model,
        encode=tok.encode,
        decode=tok.decode,
        device=dev,
        amp_dtype=amp_map[dtype_name],
        model_id=model_id,
        checkpoint=checkpoint,
        meta=dict(meta) if isinstance(meta, dict) else {},
    )
