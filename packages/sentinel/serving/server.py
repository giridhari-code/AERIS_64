"""SENTINEL inference API — model-only.

Request lifecycle (DeepInfer-style phases, single-stream):
  1. ADD      — normalize prompt + SamplingParams → Sequence (WAITING)
  2. SCHEDULE — acquire engine lock, mark RUNNING (serial scheduler)
  3. PREFILL  — forward(full prompt, state=None)
  4. DECODE   — loop: SAMPLE → APPEND → forward(token) until stop/max
  5. FINALIZE — detokenize, cut markers, FINISHED → response

No continuous batching / paged KV: one Sequence under one lock.
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentinel import __version__
from sentinel.model_executor import (
    ModelRunner,
    SamplingParams as ExecSamplingParams,
    load_model as exec_load_model,
    sample_from_logits as exec_sample_from_logits,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_runner: Optional[ModelRunner] = None
_encode = None
_decode = None
_device = torch.device("cpu")
_model_id = ""
_checkpoint_path = ""
_STOP_MARKERS = ("<|end|>", "<|user|>", "<|system|>")
_stop_ids: set[int] = set()


# ---------------------------------------------------------------------------
# Lifecycle types (DeepInfer-inspired Sequence / SamplingParams)
# ---------------------------------------------------------------------------
class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


SamplingParams = ExecSamplingParams  # model_executor


@dataclass
class Sequence:
    """One generation job — analogous to DeepInfer Sequence (single-stream)."""

    request_id: str
    prompt_text: str
    prompt_ids: List[int]
    params: SamplingParams
    status: SequenceStatus = SequenceStatus.WAITING
    output_ids: List[int] = field(default_factory=list)
    output_text: str = ""
    state: Any = None  # SENTINEL recurrent state (not paged KV)
    latency_ms: float = 0.0
    finish_reason: str = ""  # stop | length | error
    error: Optional[str] = None
    t0: float = field(default_factory=time.perf_counter)

    def append_token(self, tid: int) -> None:
        self.output_ids.append(int(tid))


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
    global _runner, _encode, _decode, _device, _model_id, _checkpoint_path, _stop_ids
    loaded = exec_load_model(checkpoint, config_path, device=device)
    stop_ids: set[int] = set()
    for marker in _STOP_MARKERS:
        try:
            ids = _to_id_list(loaded.encode(marker))
        except Exception:  # noqa: BLE001
            continue
        if len(ids) == 1:
            stop_ids.add(ids[0])
    with _lock:
        _stop_ids = stop_ids
        _runner = ModelRunner(loaded.model, loaded.device, loaded.amp_dtype)
        _encode, _decode = loaded.encode, loaded.decode
        _device = loaded.device
        _model_id = loaded.model_id
        _checkpoint_path = loaded.checkpoint
    logger.info("serve runner ready model=%s device=%s", _model_id, _device)



def _need_model() -> None:
    if _runner is None or _encode is None or _decode is None:
        raise HTTPException(status_code=503, detail="model not loaded — set NEUROFIELD_CHECKPOINT")


def _sample_from_logits(logits, params, generator):
    return exec_sample_from_logits(logits, params, generator)



def _build_prompt(user_text: str) -> str:
    t = (user_text or "").strip()
    if t.startswith("User:") or t.startswith("<|user|>") or t.startswith("System:"):
        if "Assistant:" not in t and "<|assistant|>" not in t:
            return t + "\nAssistant:"
        return t
    return f"User: {t}\nAssistant:"


# ---------------------------------------------------------------------------
# Engine step: DeepInfer-like lifecycle on one Sequence
# ---------------------------------------------------------------------------
def _forward_tokens(token_ids: List[int], state: Any) -> tuple[torch.Tensor, Any]:
    """Execute via ModelRunner → (logits V, new_state)."""
    assert _runner is not None
    if state is None:
        step = _runner.prefill(token_ids)
    elif len(token_ids) == 1:
        step = _runner.decode(token_ids[0], state)
    else:
        step = _runner.step(token_ids, state=state)
    return step.logits, step.state


def engine_run(seq: Sequence) -> Sequence:
    """
    Full lifecycle for one sequence (serial scheduler).

    ADD is done by caller; this function:
      SCHEDULE → PREFILL → (SAMPLE → APPEND → DECODE)* → FINALIZE
    """
    assert _runner is not None and _decode is not None
    seq.status = SequenceStatus.RUNNING
    gen: Optional[torch.Generator] = None
    max_new = max(1, min(int(seq.params.max_tokens), 256))

    try:
        with _lock:
            # --- SCHEDULE (serial): we hold the only worker slot ---
            if seq.params.seed is not None:
                gen = torch.Generator(
                    device=_device if _device.type == "cuda" else "cpu"
                )
                gen.manual_seed(int(seq.params.seed))

            # --- PREFILL ---
            logits, state = _forward_tokens(seq.prompt_ids, state=None)
            seq.state = state
            tid = _sample_from_logits(logits, seq.params, gen)
            if tid in _stop_ids:
                seq.finish_reason = "stop"
            else:
                seq.append_token(tid)

            # --- DECODE loop: SAMPLE already done for first; continue ---
            while (
                seq.finish_reason != "stop"
                and len(seq.output_ids) < max_new
            ):
                logits, state = _forward_tokens([seq.output_ids[-1]], seq.state)
                seq.state = state
                tid = _sample_from_logits(logits, seq.params, gen)
                if tid in _stop_ids:
                    seq.finish_reason = "stop"
                    break
                seq.append_token(tid)

            if not seq.finish_reason:
                seq.finish_reason = "length" if len(seq.output_ids) >= max_new else "stop"

        # --- FINALIZE (outside lock: detokenize) ---
        raw = (_decode(seq.output_ids) or "").strip()
        seq.output_text = _cut_at_markers(raw)
        seq.status = SequenceStatus.FINISHED
        seq.latency_ms = (time.perf_counter() - seq.t0) * 1000.0
        logger.debug(
            "lifecycle done id=%s reason=%s tokens=%d latency_ms=%.1f",
            seq.request_id,
            seq.finish_reason,
            len(seq.output_ids),
            seq.latency_ms,
        )
        return seq
    except Exception as e:  # noqa: BLE001
        seq.status = SequenceStatus.ERROR
        seq.error = f"{type(e).__name__}: {e}"
        seq.finish_reason = "error"
        seq.latency_ms = (time.perf_counter() - seq.t0) * 1000.0
        logger.exception("lifecycle error id=%s", seq.request_id)
        raise


def add_request(prompt_text: str, params: SamplingParams) -> Sequence:
    """Phase ADD: tokenize + WAITING sequence."""
    _need_model()
    assert _encode is not None
    formatted = _build_prompt(prompt_text)
    ids = _to_id_list(_encode(formatted))
    if len(ids) > 512:
        ids = ids[-512:]
    return Sequence(
        request_id=uuid.uuid4().hex[:12],
        prompt_text=formatted,
        prompt_ids=ids,
        params=params,
        status=SequenceStatus.WAITING,
    )


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
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
    request_id: str = ""
    finish_reason: str = ""


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
    finish_reason: str = ""


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
        description="Model-only SENTINEL — DeepInfer-style request lifecycle (single-stream)",
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

    def _run_chat(
        user_text: str,
        max_new: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
        session_id: Optional[str],
    ) -> ChatResponse:
        params = SamplingParams(
            max_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        try:
            seq = add_request(user_text, params)
            seq = engine_run(seq)
            return ChatResponse(
                text=seq.output_text,
                session_id=session_id or "default",
                latency_ms=seq.latency_ms,
                model=_model_id,
                request_id=seq.request_id,
                finish_reason=seq.finish_reason,
            )
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
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
            "docs": "docs/INFERENCE_ENGINE.md",
            "lifecycle": [
                "ADD",
                "SCHEDULE",
                "PREFILL",
                "DECODE_SAMPLE_LOOP",
                "FINALIZE",
            ],
            "architecture": {
                "high_level": "FastAPI → serial engine_run → SENTINEL.forward(state)",
                "main_pieces": [
                    "Sequence",
                    "SamplingParams",
                    "engine_run",
                    "SENTINEL executor",
                    "sampler",
                ],
                "model_executor": "packages/sentinel/model_executor (runner+loader)",
                "parallelism": {
                    "inference": "single_device_serial",
                    "training": "ddp_and_mesh_hooks",
                },
                "speculative_mtp_dspark": "not_implemented",
                "sampling": {
                    "temperature": True,
                    "top_p": True,
                    "top_k": True,
                    "seed": True,
                    "structured_output": "stop_markers_only",
                },
                "async_scheduling": "fastapi_async_routes_sync_generate_under_lock",
                "production_ready": "research_single_user_only",
            },
            "chat": "POST /v1/chat",
            "openai_chat": "POST /v1/chat/completions",
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
            "scheduler": "serial-lock",
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
        return _run_chat(
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
        prompt = "\n".join(parts)
        if not prompt.strip().endswith("Assistant:"):
            prompt = prompt + "\nAssistant:"
        result = _run_chat(
            prompt, req.max_tokens, req.temperature, req.top_p, 40, req.seed, None
        )
        return {
            "id": f"chatcmpl-{result.request_id}",
            "object": "chat.completion",
            "model": result.model or req.model or "aeris",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason or "stop",
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
        params = SamplingParams(
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
        )
        seq = Sequence(
            request_id=uuid.uuid4().hex[:12],
            prompt_text="",
            prompt_ids=_to_id_list(req.input_ids),
            params=params,
        )
        try:
            seq = engine_run(seq)
            return CompletionResponse(
                output_ids=seq.output_ids,
                latency_ms=seq.latency_ms,
                finish_reason=seq.finish_reason,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"generate error: {e}") from e

    return app
