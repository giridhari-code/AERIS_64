"""SENTINEL inference API — model-only.

Fixes vs early lean server:
  - top_p / top_k / seed sampling
  - OpenAI-compatible POST /v1/chat/completions
  - stop markers + text cut
  - clearer ready/root metadata
"""

from __future__ import annotations

import logging
import os
import threading
import time
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
_STOP_MARKERS = ("<|end|>", "<|user|>", "<|system|>")
_stop_ids: set[int] = set()


def _cut_at_markers(text: str) -> str:
    cut = len(text)
    for m in _STOP_MARKERS + ("\nUser:", "\nUSER:", "\nHuman:"):
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(f"NEUROFIELD_{key}") or os.environ.get(key) or default).strip()


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


def load_model(checkpoint: str, config_path: Optional[str], device: str) -> None:
    global _model, _encode, _decode, _device, _model_id, _amp_dtype, _checkpoint_path, _stop_ids
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
    stop_ids: set[int] = set()
    for marker in _STOP_MARKERS:
        try:
            ids = _to_id_list(tok.encode(marker))
        except Exception:  # noqa: BLE001
            continue
        if len(ids) == 1:
            stop_ids.add(ids[0])
    with _lock:
        _stop_ids = stop_ids
        _amp_dtype = amp[dtype_name]
        _encode, _decode = tok.encode, tok.decode
        _model = model
        _model_id = str(meta.get("model_id") or meta.get("name") or Path(checkpoint).stem or "model")
        _checkpoint_path = checkpoint
    logger.info("Loaded model %s from %s on %s", _model_id, checkpoint, _device)


def _need_model() -> None:
    if _model is None or _encode is None or _decode is None:
        raise HTTPException(status_code=503, detail="model not loaded — set NEUROFIELD_CHECKPOINT")


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: Optional[torch.Generator],
) -> int:
    """logits: (V,) float32 on device."""
    if temperature <= 1e-6:
        return int(torch.argmax(logits).item())
    logits = logits / max(float(temperature), 1e-5)
    # top-k
    k = int(top_k)
    if k > 0 and k < logits.numel():
        vals, idx = torch.topk(logits, k)
        masked = torch.full_like(logits, float("-inf"))
        masked[idx] = vals
        logits = masked
    # top-p (nucleus)
    p = float(top_p)
    if 0.0 < p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        # keep smallest prefix with cumprob >= p
        mask = cumsum > p
        # always keep first token
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    if generator is not None:
        return int(torch.multinomial(probs, 1, generator=generator).item())
    return int(torch.multinomial(probs, 1).item())


def _generate(
    input_ids: list[int],
    max_new: int,
    temperature: float,
    *,
    top_p: float = 0.9,
    top_k: int = 40,
    seed: Optional[int] = None,
) -> tuple[list[int], float]:
    assert _model is not None
    t0 = time.perf_counter()
    ids = _to_id_list(input_ids)
    max_new = max(1, min(int(max_new), 256))
    new_tokens: List[int] = []
    gen: Optional[torch.Generator] = None
    with _lock:
        model = _model
        device = _device
        amp = _amp_dtype
        if seed is not None:
            gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
            gen.manual_seed(int(seed))
        state = None
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
                    write_slow=True,
                )
            state = out.state
            logits = out.logits[0, -1].float()
            nid = _sample_from_logits(
                logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
            )
            if nid in _stop_ids:
                return new_tokens, (time.perf_counter() - t0) * 1000.0
            new_tokens.append(nid)
            for _ in range(max_new - 1):
                x = torch.tensor([[nid]], dtype=torch.long, device=device)
                with cm:
                    out = model(
                        x,
                        state=state,
                        targets=None,
                        collect_stats=False,
                        write_slow=True,
                    )
                state = out.state
                logits = out.logits[0, -1].float()
                nid = _sample_from_logits(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
                )
                if nid in _stop_ids:
                    break
                new_tokens.append(nid)
    return new_tokens, (time.perf_counter() - t0) * 1000.0


def _build_prompt(user_text: str) -> str:
    t = (user_text or "").strip()
    # Already role-formatted
    if t.startswith("User:") or t.startswith("<|user|>"):
        if "Assistant:" not in t and "<|assistant|>" not in t:
            return t + "\nAssistant:"
        return t
    return f"User: {t}\nAssistant:"


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(64, ge=1, le=256)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0, le=1000)
    seed: Optional[int] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    session_id: str
    latency_ms: float
    model: str


class CompletionRequest(BaseModel):
    input_ids: list[int] = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(48, ge=1, le=256)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0, le=1000)
    seed: Optional[int] = None


class CompletionResponse(BaseModel):
    output_ids: list[int]
    latency_ms: float


class OAMessage(BaseModel):
    role: str
    content: str


class OAChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OAMessage]
    max_tokens: int = Field(64, ge=1, le=256)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    seed: Optional[int] = None


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
        description="Model-only SENTINEL serve (top_p/top_k + OpenAI chat.completions)",
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

    def _chat_core(
        user_text: str,
        max_new: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
        session_id: Optional[str],
    ) -> ChatResponse:
        _need_model()
        assert _encode is not None and _decode is not None
        try:
            text_in = _build_prompt(user_text)
            ids = _to_id_list(_encode(text_in))
            if len(ids) > 512:
                ids = ids[-512:]
            new_ids, lat = _generate(
                ids,
                max_new,
                temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
            text = _cut_at_markers((_decode(new_ids) or "").strip())
            return ChatResponse(
                text=text,
                session_id=session_id or "default",
                latency_ms=lat,
                model=_model_id,
            )
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("chat generate failed")
            raise HTTPException(
                status_code=500,
                detail=f"generate error: {type(e).__name__}: {e}",
            ) from e

    @app.get("/")
    def root():
        return {
            "product": product,
            "version": __version__,
            "mode": "model-only",
            "chat": "POST /v1/chat",
            "openai_chat": "POST /v1/chat/completions",
            "ready": _model is not None,
            "note": "Gibberish output ⇒ need more pretrain/SFT data; engine is OK if ready=true",
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
                    "id": _model_id or "aeris",
                    "object": "model",
                    "owned_by": "sentinel",
                    "ready": _model is not None,
                }
            ],
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        return _chat_core(
            req.prompt,
            req.max_new_tokens,
            req.temperature,
            req.top_p,
            req.top_k,
            req.seed,
            req.session_id,
        )

    @app.post("/v1/chat/completions")
    def chat_completions(req: OAChatRequest):
        # OpenAI-style: stitch messages into one prompt
        parts: List[str] = []
        for m in req.messages:
            role = (m.role or "user").lower()
            content = (m.content or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        if not parts:
            raise HTTPException(status_code=400, detail="empty messages")
        # Ensure model continues as assistant
        prompt = "\n".join(parts)
        if not prompt.strip().endswith("Assistant:"):
            prompt = prompt + "\nAssistant:"
        result = _chat_core(
            prompt,
            req.max_tokens,
            req.temperature,
            req.top_p,
            40,
            req.seed,
            None,
        )
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": result.model or req.model or "aeris",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "latency_ms": result.latency_ms,
        }

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(req: CompletionRequest):
        _need_model()
        try:
            new_ids, lat = _generate(
                req.input_ids,
                req.max_new_tokens,
                req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
            )
            return CompletionResponse(output_ids=new_ids, latency_ms=lat)
        except Exception as e:  # noqa: BLE001
            logger.exception("completions failed")
            raise HTTPException(status_code=500, detail=f"generate error: {e}") from e

    return app
