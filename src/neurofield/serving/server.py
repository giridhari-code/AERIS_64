"""NeuroField inference API (FastAPI).

Features
  - Chat + raw-token completions, SSE streaming
  - Optional API-key auth (NEUROFIELD_API_KEYS, header only)
  - Rate limiting, Prometheus metrics, health/readiness
  - Safe, thread-safe session handling (TTL, max sessions, owner binding)
  - Architecture is inferred from the weights, so empty/wrong config.json
    can no longer produce a silently mis-shaped (or half-random) model.

Environment (NEUROFIELD_* wins; the short names are kept for compatibility)
  NEUROFIELD_CHECKPOINT | CHECKPOINT        folder / .safetensors / legacy .pt
  NEUROFIELD_CONFIG     | CONFIG            optional YAML (training/safety knobs)
  NEUROFIELD_DEVICE     | DEVICE            cpu | cuda | cuda:0
  NEUROFIELD_DTYPE                          float32 (default) | bfloat16 | float16 (autocast)
  NEUROFIELD_STRICT_LOAD                    1 (default): refuse missing weights
  NEUROFIELD_SESSION_TTL                    seconds, default 1800
  NEUROFIELD_MAX_SESSIONS                   default 256
  NEUROFIELD_MAX_CONTEXT                    max prompt tokens, default = model max_seq_len
  NEUROFIELD_PRODUCT                        display name, default "NeuroField"
  NEUROFIELD_METRICS_PUBLIC                 1 (default) / 0 => /metrics needs key
  NEUROFIELD_LOG_JSON / NEUROFIELD_LOG_LEVEL  structured logs (default off / INFO)
  NEUROFIELD_TRUST_PROXY                    1 => rate-limit by X-Forwarded-For
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from neurofield import __version__
from neurofield.checkpoint.resolve import load_model_from_checkpoint
from neurofield.config import NeuroFieldConfig
from neurofield.model import NeuroField
from neurofield.safety import SafetyMonitor
from neurofield.serving.adaptive_length import adaptive_sampling, should_stop_generation
from neurofield.serving.auth import client_ip, rate_limit_from_env, require_api_key
from neurofield.serving.chat_history import append_turn, build_prompt, reset_history, set_use_specials
from neurofield.serving.reply_polish import polish_reply
from neurofield.utils.logging import setup_production_logging
from neurofield.utils.text_norm import normalize_prompt

logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_STOP_TAIL_TOKENS = 64  # only decode the tail when checking stop conditions
_stop_token_ids: set[int] = set()  # <|end|>, <|user|> from tokenizer when present


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(f"NEUROFIELD_{name}") or os.environ.get(name) or default


# --------------------------------------------------------------------------
# Global (per-process) serving state. One model, guarded by _lock.
# --------------------------------------------------------------------------
# Plain Lock (not RLock): a streaming generator may resume on a different
# worker thread, and only a plain Lock can be released from another thread.
_lock = threading.Lock()  # serialises model use + session table mutation
_stats_lock = threading.Lock()
_stats = {"requests": 0, "errors": 0, "latency_ms": 0.0}

_model: Optional[NeuroField] = None
_cfg: Optional[NeuroFieldConfig] = None
_device: torch.device = torch.device("cpu")
_monitor: Optional[SafetyMonitor] = None
_encode = None
_decode = None
_model_id = ""
_checkpoint_path = ""
_max_context = 2048
_amp_dtype: Optional[torch.dtype] = None
_limiter = rate_limit_from_env()

# sid -> {"state": Any, "owner": Optional[str], "ts": float}
_sessions: dict[str, dict[str, Any]] = {}


def _bump(key: str, value: float = 1.0) -> None:
    with _stats_lock:
        _stats[key] += value


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class CompletionRequest(BaseModel):
    input_ids: list[int] = Field(..., min_length=1, max_length=8192)
    max_new_tokens: int = Field(48, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    session_id: Optional[str] = None
    reset_session: bool = False


class CompletionResponse(BaseModel):
    output_ids: list[int]
    session_id: str
    latency_ms: float
    audit: Optional[dict[str, Any]] = None


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(48, ge=1, le=8192, description="Cap; with auto_length, acts as maximum")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="0 = greedy")
    auto_length: bool = Field(True, description="Dynamic max_tokens + temperature (Grok-style)")
    postprocess: bool = Field(
        True,
        description="Apply reply polish + first-line cut. Set false to see the raw model output.",
    )
    session_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    text: str
    session_id: str
    latency_ms: float
    model: str = ""


# --------------------------------------------------------------------------
# Model loading (shared logic lives in neurofield.checkpoint.resolve)
# --------------------------------------------------------------------------
def load_model(checkpoint: str, config_path: Optional[str], device: str) -> None:
    global _model, _cfg, _device, _monitor, _model_id, _checkpoint_path, _max_context, _amp_dtype, _stop_token_ids
    global _encode, _decode
    want = device or "cpu"
    _device = torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")
    _checkpoint_path = checkpoint

    model, tok, full_cfg, meta = load_model_from_checkpoint(
        checkpoint,
        config_path,
        device=str(_device),
        strict=_env("STRICT_LOAD", "1") == "1",
    )

    # Weights stay fp32; reduced precision is applied with autocast so the
    # recurrent state tensors (created fp32) never mismatch the weights.
    dtype_name = _env("DTYPE", "float32")
    amp = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in amp:
        raise ValueError(f"Unknown NEUROFIELD_DTYPE={dtype_name}")
    # Server: never share a running-surprise baseline between sessions.
    model.meta.track_global = False

    with _lock:
        _amp_dtype = amp[dtype_name]
        _encode, _decode = tok.encode, tok.decode
        _stop_token_ids = set(getattr(tok, "stop_ids", set()) or set())
        set_use_specials(bool(_stop_token_ids))
        _model = model
        _cfg = full_cfg
        _sessions.clear()
        _monitor = SafetyMonitor(z_threshold=full_cfg.safety.anomaly_z_threshold, log_path=None)
        _model_id = str(meta.get("model_id") or meta.get("name") or Path(checkpoint).stem or "model")
        _max_context = int(_env("MAX_CONTEXT", str(full_cfg.model.max_seq_len)) or full_cfg.model.max_seq_len)
    logger.info(
        "Model loaded on %s (%s) | path=%s | id=%s | params=%s | vocab=%s | stop_ids=%s",
        _device,
        dtype_name,
        checkpoint,
        _model_id,
        model.param_report()["total"],
        tok.vocab_size,
        sorted(_stop_token_ids),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _model
    with _lock:
        _model = None
        _sessions.clear()
    logger.info("Server shutdown complete")


# --------------------------------------------------------------------------
# Session table
# --------------------------------------------------------------------------
def _valid_session_id(sid: Optional[str]) -> str:
    if sid is None:
        return uuid.uuid4().hex
    if not _SESSION_ID_RE.match(sid):
        raise HTTPException(status_code=422, detail="session_id must match [A-Za-z0-9_-]{8,64}")
    return sid


def _evict_locked(now: float) -> None:
    ttl = float(_env("SESSION_TTL", "1800") or 1800)
    max_s = int(_env("MAX_SESSIONS", "256") or 256)
    for sid in [s for s, v in _sessions.items() if now - v["ts"] > ttl]:
        _sessions.pop(sid, None)
        reset_history(sid)
    if len(_sessions) > max_s:
        for sid, _ in sorted(_sessions.items(), key=lambda kv: kv[1]["ts"])[: len(_sessions) - max_s]:
            _sessions.pop(sid, None)
            reset_history(sid)


def _claim_session_locked(sid: str, owner: Optional[str]) -> dict[str, Any]:
    now = time.time()
    _evict_locked(now)
    sess = _sessions.get(sid)
    if sess is not None and owner is not None and sess["owner"] not in (None, owner):
        raise HTTPException(status_code=403, detail="session belongs to another key")
    if sess is None:
        sess = {"state": None, "owner": owner, "ts": now}
        _sessions[sid] = sess
    sess["ts"] = now
    return sess


# --------------------------------------------------------------------------
# Generation core
# --------------------------------------------------------------------------
def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 1e-5:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _stop_now(generated: list[int], prompt_len: int) -> bool:
    """Stop on special token IDs when available; else text heuristics."""
    global _stop_token_ids
    cont = generated[prompt_len:]
    if not cont:
        return False
    if _stop_token_ids and cont[-1] in _stop_token_ids:
        return True
    tail = cont[-( _STOP_TAIL_TOKENS) :]
    try:
        partial = _decode(tail) if _decode else ""
    except Exception:
        return False
    return bool(partial) and should_stop_generation(partial)



def _token_stream(
    input_ids: list[int],
    max_new: int,
    temperature: float,
    sid: str,
    owner: Optional[str],
    reset: bool,
    want_audit: bool = False,
) -> Iterator[tuple[int, Optional[dict]]]:
    """Yield (next_id, audit) while holding the model lock. Persists session state at the end."""
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    input_ids = input_ids[-_max_context:]
    with _lock:
        sess = _claim_session_locked(sid, owner)
        if reset:
            sess["state"] = None
        state = sess["state"]
        cur = torch.tensor([input_ids], dtype=torch.long, device=_device)
        generated = list(input_ids)
        prompt_len = len(input_ids)
        try:
            amp_ctx = (
                torch.autocast(device_type=_device.type, dtype=_amp_dtype)
                if _amp_dtype is not None
                else contextlib.nullcontext()
            )
            with torch.no_grad(), amp_ctx:
                for _ in range(max_new):
                    out = _model(cur, state=state, return_audit=True)
                    state = out.state
                    audit = out.audit
                    if _monitor is not None and audit:
                        alarms = _monitor.update(audit)
                        if (
                            alarms
                            and _cfg is not None
                            and _cfg.safety.freeze_on_anomaly
                            and _monitor.should_freeze()
                        ):
                            _model.freeze_writes()
                            logger.warning("Anomaly freeze: %s", alarms)
                    nid = _sample(out.logits[:, -1, :], temperature)
                    generated.append(nid)
                    cur = torch.tensor([[nid]], dtype=torch.long, device=_device)
                    yield nid, (audit if want_audit else None)
                    if _stop_now(generated, prompt_len):
                        break
        finally:
            sess["state"] = state


def _generate(
    input_ids: list[int],
    max_new: int,
    temperature: float,
    sid: str,
    owner: Optional[str],
    reset: bool,
    want_audit: bool = False,
) -> tuple[list[int], float, Optional[dict]]:
    start = time.perf_counter()
    new_ids: list[int] = []
    audit = None
    for nid, a in _token_stream(input_ids, max_new, temperature, sid, owner, reset, want_audit):
        new_ids.append(nid)
        audit = a or audit
    lat = (time.perf_counter() - start) * 1000
    _bump("latency_ms", lat)
    return new_ids, lat, audit


def _postprocess(prompt: str, raw: str, enabled: bool) -> str:
    if not enabled:
        return raw.strip()
    text = polish_reply(prompt, raw)
    if "\n" in text:
        first = text.split("\n")[0].strip()
        if len(first) >= 8:
            text = first
    return text


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------

# Sandbox tool execution (optional; enabled unless NEUROFIELD_SANDBOX=0)
from neurofield.sandbox import SandboxConfig, SandboxRegistry

_sandbox_registry: SandboxRegistry | None = None


def _sandbox() -> SandboxRegistry:
    global _sandbox_registry
    if _sandbox_registry is None:
        _sandbox_registry = SandboxRegistry(
            SandboxConfig(
                timeout_sec=float(__import__("os").environ.get("NEUROFIELD_SANDBOX_TIMEOUT", "8")),
                allow_network=__import__("os").environ.get("NEUROFIELD_SANDBOX_NETWORK", "0") == "1",
            ),
            max_sessions=int(__import__("os").environ.get("NEUROFIELD_SANDBOX_MAX_SESSIONS", "64")),
            max_calls_per_session=int(__import__("os").environ.get("NEUROFIELD_SANDBOX_MAX_CALLS", "32")),
        )
    return _sandbox_registry


def create_app(
    checkpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> FastAPI:
    # LOG_JSON / LOG_LEVEL were documented in .env/Docker but never wired; INFO logs were invisible.
    setup_production_logging(
        level=_env("LOG_LEVEL", "INFO") or "INFO", json_logs=_env("LOG_JSON", "0") == "1"
    )
    checkpoint = checkpoint or _env("CHECKPOINT")
    config_path = config_path or _env("CONFIG")
    device = _env("DEVICE", device) or "cpu"
    product = _env("PRODUCT", "NeuroField") or "NeuroField"

    app = FastAPI(
        title=f"{product} API",
        description=f"{product} — NeuroField two-speed memory architecture.",
        version=__version__,
        lifespan=lifespan,
    )

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    if checkpoint:
        load_model(checkpoint, config_path, device)

    def _owner(api_key: Optional[str]) -> Optional[str]:
        # Session ownership is enforced only when auth is enabled.
        return api_key

    def _limit(request: Request, api_key: Optional[str]) -> None:
        _limiter.check(api_key or client_ip(request))

    def _need_model() -> None:
        if _model is None or _encode is None or _decode is None:
            raise HTTPException(status_code=503, detail="model not loaded")

    async def _metrics_guard(request: Request) -> None:
        if _env("METRICS_PUBLIC", "1") == "1":
            return
        await require_api_key(request.headers.get("x-api-key"))

    @app.get("/")
    async def frontend():
        index = static_dir / "index.html"
        if index.is_file():
            return FileResponse(str(index))
        return PlainTextResponse(f"{product} API. See /docs")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "product": product, "version": __version__}

    @app.get("/readyz")
    async def readyz():
        if _model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {
            "status": "ready",
            "device": str(_device),
            "model": _model_id,
            "checkpoint": _checkpoint_path,
            "has_tokenizer": _encode is not None,
            "params": _model.param_report()["total"],
        }

    @app.get("/metrics", dependencies=[Depends(_metrics_guard)])
    async def metrics():
        with _stats_lock:
            req, err, lat = _stats["requests"], _stats["errors"], _stats["latency_ms"]
        body = (
            "# HELP neurofield_requests_total Total requests\n"
            "# TYPE neurofield_requests_total counter\n"
            f"neurofield_requests_total {int(req)}\n"
            "# HELP neurofield_errors_total Total errors\n"
            "# TYPE neurofield_errors_total counter\n"
            f"neurofield_errors_total {int(err)}\n"
            "# HELP neurofield_latency_ms Average latency\n"
            "# TYPE neurofield_latency_ms gauge\n"
            f"neurofield_latency_ms {lat / max(1, req):.2f}\n"
            "# HELP neurofield_sessions Active sessions\n"
            "# TYPE neurofield_sessions gauge\n"
            f"neurofield_sessions {len(_sessions)}\n"
        )
        return PlainTextResponse(body)

    @app.get("/v1/models")
    async def list_models(_key: Optional[str] = Depends(require_api_key)):
        return {
            "object": "list",
            "data": [
                {
                    "id": _model_id,
                    "object": "model",
                    "owned_by": "neurofield",
                    "ready": _model is not None,
                }
            ],
        }

    # NOTE: inference endpoints are plain `def` so FastAPI runs them in its
    # threadpool; the event loop is never blocked by torch.

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(
        req: CompletionRequest,
        request: Request,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        _limit(request, api_key)
        _need_model()
        _bump("requests")
        sid = _valid_session_id(req.session_id)
        try:
            new_ids, lat, audit = _generate(
                req.input_ids, req.max_new_tokens, req.temperature, sid,
                _owner(api_key), req.reset_session, want_audit=True,
            )
            return CompletionResponse(
                output_ids=list(req.input_ids) + new_ids, session_id=sid, latency_ms=lat, audit=audit
            )
        except HTTPException:
            raise
        except Exception as e:
            _bump("errors")
            logger.exception("completions failed")
            raise HTTPException(status_code=500, detail="internal error") from e

    def _prepare_chat(req: ChatRequest, sid: str) -> tuple[str, list[int], int, float]:
        prompt = normalize_prompt(req.prompt)
        input_ids = _encode(build_prompt(sid, prompt)) or _encode(prompt) or [0]
        max_new, temperature = adaptive_sampling(
            prompt,
            user_max=req.max_new_tokens,
            user_temp=req.temperature,
            auto=req.auto_length,
        )
        return prompt, input_ids, max_new, temperature

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(
        req: ChatRequest,
        request: Request,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        _limit(request, api_key)
        _need_model()
        _bump("requests")
        sid = _valid_session_id(req.session_id)
        try:
            prompt, input_ids, max_new, temperature = _prepare_chat(req, sid)
            # History is re-fed as text every turn, so start from a fresh recurrent
            # state (reset=True); otherwise earlier turns would be seen twice.
            new_ids, lat, _ = _generate(
                input_ids, max_new, temperature, sid, _owner(api_key), reset=True
            )
            text = _postprocess(prompt, _decode(new_ids), req.postprocess)
            append_turn(sid, "user", prompt)
            append_turn(sid, "assistant", text)
            return ChatResponse(text=text, session_id=sid, latency_ms=lat, model=_model_id)
        except HTTPException:
            raise
        except Exception as e:
            _bump("errors")
            logger.exception("chat failed")
            raise HTTPException(status_code=500, detail="internal error") from e

    @app.post("/v1/chat/stream")
    def chat_stream(
        req: ChatRequest,
        request: Request,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        _limit(request, api_key)
        _need_model()
        _bump("requests")
        sid = _valid_session_id(req.session_id)
        prompt, input_ids, max_new, temperature = _prepare_chat(req, sid)
        owner = _owner(api_key)

        def event_gen() -> Iterator[str]:
            start = time.perf_counter()
            new_ids: list[int] = []
            yield f"data: {json.dumps({'session_id': sid, 'event': 'start'})}\n\n"
            try:
                for nid, _ in _token_stream(input_ids, max_new, temperature, sid, owner, reset=True):
                    new_ids.append(nid)
                    yield f"data: {json.dumps({'token': _decode([nid]), 'id': nid})}\n\n"
            except HTTPException as e:
                yield f"data: {json.dumps({'event': 'error', 'detail': e.detail})}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception:
                _bump("errors")
                logger.exception("stream failed")
                yield f"data: {json.dumps({'event': 'error', 'detail': 'internal error'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            lat = (time.perf_counter() - start) * 1000
            _bump("latency_ms", lat)
            text = _postprocess(prompt, _decode(new_ids), req.postprocess)
            append_turn(sid, "user", prompt)
            append_turn(sid, "assistant", text)
            yield f"data: {json.dumps({'event': 'done', 'latency_ms': lat, 'text': text})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.post("/v1/reset")
    def reset_session(session_id: str, api_key: Optional[str] = Depends(require_api_key)):
        if not _SESSION_ID_RE.match(session_id):
            raise HTTPException(status_code=422, detail="invalid session_id")
        with _lock:
            sess = _sessions.get(session_id)
            owner = _owner(api_key)
            if sess is not None and owner is not None and sess["owner"] not in (None, owner):
                raise HTTPException(status_code=403, detail="session belongs to another key")
            _sessions.pop(session_id, None)
        reset_history(session_id)  # previously never called => history survived "reset"
        if __import__("os").environ.get("NEUROFIELD_SANDBOX", "1") != "0":
            _sandbox().drop(session_id)
        return {"status": "reset", "session_id": session_id}

    # ------------------------------------------------------------------
    # Sandbox tools (production) — disabled when NEUROFIELD_SANDBOX=0
    # ------------------------------------------------------------------
    if __import__("os").environ.get("NEUROFIELD_SANDBOX", "1") != "0":
        from pydantic import BaseModel, Field as PydField

        class _ShellReq(BaseModel):
            command: str = PydField(..., min_length=1, max_length=4000)
            session_id: str = PydField(default="default", max_length=64)

        class _PyReq(BaseModel):
            code: str = PydField(..., min_length=1, max_length=20_000)
            session_id: str = PydField(default="default", max_length=64)

        class _ToolResetReq(BaseModel):
            session_id: str = PydField(default="default", max_length=64)

        @app.post("/v1/tools/shell")
        def tools_shell(req: _ShellReq, api_key: Optional[str] = Depends(require_api_key)):
            sid = _valid_session_id(req.session_id)
            result = _sandbox().get(sid).run_shell(req.command)
            return result.to_dict()

        @app.post("/v1/tools/python")
        def tools_python(req: _PyReq, api_key: Optional[str] = Depends(require_api_key)):
            sid = _valid_session_id(req.session_id)
            result = _sandbox().get(sid).run_python(req.code)
            return result.to_dict()

        @app.get("/v1/tools/audit")
        def tools_audit(session_id: str = "default", api_key: Optional[str] = Depends(require_api_key)):
            sid = _valid_session_id(session_id)
            sb = _sandbox().get(sid)
            return {"session_id": sid, "calls": sb.calls, "audit": sb.audit()}

        @app.post("/v1/tools/reset")
        def tools_reset(req: _ToolResetReq, api_key: Optional[str] = Depends(require_api_key)):
            sid = _valid_session_id(req.session_id)
            _sandbox().reset(sid)
            return {"status": "reset", "session_id": sid}

    return app


# uvicorn entry: neurofield.serving.server:create_app --factory
