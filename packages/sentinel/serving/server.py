"""SENTINEL inference API — model-only (no tools / agent / RAG / MCP / buddy).

uvicorn sentinel.api.app:create_app --factory
Env: NEUROFIELD_CHECKPOINT, NEUROFIELD_DEVICE, NEUROFIELD_DTYPE
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
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
_sessions: dict[str, Any] = {}


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(f"NEUROFIELD_{key}") or os.environ.get(key) or default).strip()


def load_model(checkpoint: str, config_path: Optional[str], device: str) -> None:
    global _model, _encode, _decode, _device, _model_id, _amp_dtype, _sessions
    want = device or "cpu"
    _device = torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")
    model, tok, full_cfg, meta = load_model_from_checkpoint(
        checkpoint,
        config_path,
        device=str(_device),
        strict=_env("STRICT_LOAD", "1") == "1",
    )
    dtype_name = _env("DTYPE", "float32")
    amp = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in amp:
        raise ValueError(f"Unknown NEUROFIELD_DTYPE={dtype_name}")
    if hasattr(model, "meta") and model.meta is not None:
        model.meta.track_global = False
    with _lock:
        _amp_dtype = amp[dtype_name]
        _encode, _decode = tok.encode, tok.decode
        _model = model
        _sessions.clear()
        _model_id = str(meta.get("model_id") or meta.get("name") or Path(checkpoint).stem or "model")
    logger.info("Loaded model %s from %s on %s", _model_id, checkpoint, _device)


def _need_model() -> None:
    if _model is None or _encode is None or _decode is None:
        raise HTTPException(status_code=503, detail="model not loaded — set NEUROFIELD_CHECKPOINT")


def _generate(input_ids: list[int], max_new: int, temperature: float) -> tuple[list[int], float]:
    assert _model is not None and _encode is not None
    t0 = time.perf_counter()
    ids = list(input_ids)
    max_new = max(1, min(int(max_new), 512))
    temperature = float(temperature)
    with _lock:
        model = _model
        device = _device
        amp = _amp_dtype
        state = None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new):
            ctx = {
                "device_type": device.type,
                "dtype": amp,
                "enabled": amp is not None and device.type == "cuda",
            }
            with torch.inference_mode():
                with torch.autocast(**ctx) if ctx["enabled"] else torch.nullcontext():
                    out = model(x, state=state)
            state = out.state
            logits = out.logits[0, -1].float()
            if temperature <= 0:
                nid = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                nid = int(torch.multinomial(probs, 1).item())
            ids.append(nid)
            x = torch.tensor([[nid]], dtype=torch.long, device=device)
    return ids[len(input_ids) :], (time.perf_counter() - t0) * 1000.0


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    session_id: str
    latency_ms: float
    model: str


class CompletionRequest(BaseModel):
    input_ids: list[int] = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(48, ge=1, le=512)
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
        description="Model-only SENTINEL serve (no tools/agent/RAG/MCP).",
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
        logger.warning("No NEUROFIELD_CHECKPOINT — chat will 503 until set")

    @app.get("/")
    def root():
        return {
            "product": product,
            "version": __version__,
            "mode": "model-only",
            "chat": "POST /v1/chat",
            "completions": "POST /v1/completions",
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
            "checkpoint": checkpoint,
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
        # Simple chat format (matches many pretrain text styles)
        text_in = f"User: {prompt}\nAssistant:"
        ids = _encode(text_in) or [0]
        # context trim
        if len(ids) > 512:
            ids = ids[-512:]
        new_ids, lat = _generate(ids, req.max_new_tokens, req.temperature)
        text = (_decode(new_ids) or "").strip()
        sid = req.session_id or "default"
        return ChatResponse(text=text, session_id=sid, latency_ms=lat, model=_model_id)

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(req: CompletionRequest):
        _need_model()
        new_ids, lat = _generate(req.input_ids, req.max_new_tokens, req.temperature)
        return CompletionResponse(output_ids=new_ids, latency_ms=lat)

    # Explicit 410 for removed surfaces (clearer than generic 404)
    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def removed(full_path: str, request: Request):
        path = "/" + full_path
        if path in {"/", "/healthz", "/health", "/readyz", "/v1/models", "/v1/chat", "/v1/completions"}:
            raise HTTPException(status_code=404, detail="not found")
        if path.startswith("/v1/") or path.startswith("/ag-ui"):
            raise HTTPException(
                status_code=410,
                detail="removed — model-only server (use POST /v1/chat)",
            )
        raise HTTPException(status_code=404, detail="not found")

    return app
