"""SENTINEL inference API — model-only (no tools / agent / RAG / MCP)."""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentinel import __version__
from sentinel.checkpoint.resolve import load_model_from_checkpoint
from sentinel.model import SENTINEL

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Optional[SENTINEL] = None
_encode = None
_decode = None
_device = torch.device("cpu")
_model_id = ""
_amp_dtype: Optional[torch.dtype] = None
_checkpoint_path = ""


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(f"NEUROFIELD_{key}") or os.environ.get(key) or default).strip()


def load_model(checkpoint: str, config_path: Optional[str], device: str) -> None:
    global _model, _encode, _decode, _device, _model_id, _amp_dtype, _checkpoint_path
    want = device or "cpu"
    _device = torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")
    model, tok, _full_cfg, meta = load_model_from_checkpoint(
        checkpoint,
        config_path,
        device=str(_device),
        strict=_env("STRICT_LOAD", "1") == "1",
    )
    model.eval()
    dtype_name = _env("DTYPE", "float32")
    amp = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in amp:
        raise ValueError(f"Unknown NEUROFIELD_DTYPE={dtype_name}")
    if hasattr(model, "meta") and model.meta is not None:
        try:
            model.meta.track_global = False
        except Exception:
            pass
    with _lock:
        _amp_dtype = amp[dtype_name]
        _encode, _decode = tok.encode, tok.decode
        _model = model
        _model_id = str(meta.get("model_id") or meta.get("name") or Path(checkpoint).stem or "model")
        _checkpoint_path = checkpoint
    logger.info("Loaded model %s from %s on %s", _model_id, checkpoint, _device)


def _need_model() -> None:
    if _model is None or _encode is None or _decode is None:
        raise HTTPException(status_code=503, detail="model not loaded — set NEUROFIELD_CHECKPOINT")


def _to_id_list(ids: Any) -> List[int]:
    if ids is None:
        return [0]
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.view(-1).tolist()]
    if isinstance(ids, (list, tuple)):
        out: List[int] = []
        for x in ids:
            if isinstance(x, (list, tuple)):
                out.extend(int(y) for y in x)
            else:
                out.append(int(x))
        return out or [0]
    return [int(ids)]


def _generate(input_ids: list[int], max_new: int, temperature: float) -> tuple[list[int], float]:
    assert _model is not None
    t0 = time.perf_counter()
    ids = _to_id_list(input_ids)
    max_new = max(1, min(int(max_new), 256))
    temperature = float(temperature)
    new_tokens: List[int] = []
    with _lock:
        model = _model
        device = _device
        amp = _amp_dtype
        state = None
        # Prefill full prompt once
        x = torch.tensor([ids], dtype=torch.long, device=device)
        use_amp = amp is not None and device.type == "cuda"
        cm = torch.autocast(device_type=device.type, dtype=amp) if use_amp else nullcontext()
        with torch.inference_mode():
            with cm:
                out = model(
                    x,
                    state=state,
                    targets=None,
                    collect_stats=False,
                    write_slow=False,
                )
            state = out.state
            logits = out.logits[0, -1].float()
            if temperature <= 1e-6:
                nid = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                nid = int(torch.multinomial(probs, 1).item())
            new_tokens.append(nid)
            # Decode step-by-step
            for _ in range(max_new - 1):
                x = torch.tensor([[nid]], dtype=torch.long, device=device)
                with cm:
                    out = model(
                        x,
                        state=state,
                        targets=None,
                        collect_stats=False,
                        write_slow=False,
                    )
                state = out.state
                logits = out.logits[0, -1].float()
                if temperature <= 1e-6:
                    nid = int(torch.argmax(logits).item())
                else:
                    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                    nid = int(torch.multinomial(probs, 1).item())
                new_tokens.append(nid)
    return new_tokens, (time.perf_counter() - t0) * 1000.0


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(64, ge=1, le=256)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    session_id: str
    latency_ms: float
    model: str


class CompletionRequest(BaseModel):
    input_ids: list[int] = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(48, ge=1, le=256)
    temperature: float = Field(1.0, ge=0.0, le=2.0)


class CompletionResponse(BaseModel):
    output_ids: list[int]
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app(
    checkpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> FastAPI:
    checkpoint = checkpoint or _env("CHECKPOINT")
    config_path = config_path or _env("CONFIG") or None
    device = _env("DEVICE", device) or "cpu"
    product = _env("PRODUCT", "SENTINEL") or "SENTINEL"

    if not checkpoint:
        for candidate in (
            Path("checkpoints/examples/AERIS_main"),
            Path(__file__).resolve().parents[3] / "checkpoints" / "examples" / "AERIS_main",
        ):
            if (candidate / "model.safetensors").is_file():
                checkpoint = str(candidate)
                break

    app = FastAPI(
        title=f"{product} API",
        description="Model-only SENTINEL serve",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if checkpoint:
        try:
            load_model(checkpoint, config_path, device)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to load checkpoint %s: %s", checkpoint, e)
    else:
        logger.warning("No NEUROFIELD_CHECKPOINT")

    @app.get("/")
    def root():
        return {
            "product": product,
            "version": __version__,
            "mode": "model-only",
            "chat": "POST /v1/chat",
            "ready": _model is not None,
        }

    @app.get("/healthz")
    @app.get("/health")
    def healthz():
        return {"status": "ok", "product": product, "version": __version__}

    @app.get("/readyz")
    def readyz():
        if _model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {
            "status": "ready",
            "device": str(_device),
            "model": _model_id,
            "checkpoint": _checkpoint_path,
            "has_tokenizer": _encode is not None,
            "mode": "model-only",
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": _model_id or "",
                    "object": "model",
                    "owned_by": "sentinel",
                    "ready": _model is not None,
                }
            ],
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        _need_model()
        assert _encode is not None and _decode is not None
        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="empty prompt")
        try:
            text_in = f"User: {prompt}\nAssistant:"
            raw_ids = _encode(text_in)
            ids = _to_id_list(raw_ids)
            if len(ids) > 512:
                ids = ids[-512:]
            new_ids, lat = _generate(ids, req.max_new_tokens, req.temperature)
            text = (_decode(new_ids) or "").strip()
            sid = req.session_id or "default"
            return ChatResponse(text=text, session_id=sid, latency_ms=lat, model=_model_id)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("chat generate failed")
            raise HTTPException(
                status_code=500,
                detail=f"generate error: {type(e).__name__}: {e}",
            ) from e

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(req: CompletionRequest):
        _need_model()
        try:
            new_ids, lat = _generate(req.input_ids, req.max_new_tokens, req.temperature)
            return CompletionResponse(output_ids=new_ids, latency_ms=lat)
        except Exception as e:  # noqa: BLE001
            logger.exception("completions failed")
            raise HTTPException(status_code=500, detail=f"generate error: {e}") from e

    return app
