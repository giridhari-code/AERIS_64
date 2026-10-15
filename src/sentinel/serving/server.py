"""SENTINEL inference API (FastAPI).

Designed for inference-at-scale: the underlying two-speed memory (fast delta-rule
matrix + slow ring) has **constant size** and does not suffer the linear memory
growth of static KV caching. Session state stays bounded even for long-running
or highly concurrent workloads.

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
  NEUROFIELD_PRODUCT                        display name, default "SENTINEL"
  NEUROFIELD_METRICS_PUBLIC                 1 (default) / 0 => /metrics needs key
  NEUROFIELD_LOG_JSON / NEUROFIELD_LOG_LEVEL  structured logs (default off / INFO)
  NEUROFIELD_TRUST_PROXY                    1 => rate-limit by X-Forwarded-For
"""

from __future__ import annotations

import contextlib
import hashlib
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

from sentinel import __version__
from sentinel.checkpoint.resolve import load_model_from_checkpoint
from sentinel.config import SENTINELConfig
from sentinel.model import SENTINEL
from sentinel.safety import SafetyMonitor
from sentinel.serving.adaptive_length import adaptive_sampling, should_stop_generation
from sentinel.serving.auth import (
    client_ip,
    get_api_keys,
    rate_limit_from_env,
    require_api_key,
    require_api_key_strict,
)

from sentinel.serving.chat_history import (
    append_turn,
    build_prompt,
    build_user_part,
    history_len,
    reset_history,
    restore_turns,
    set_system_prompt,
    default_system_prompt,
    get_system_prompt,
    set_use_specials,
)

from sentinel.serving.user_memory import UserMemoryStore, cap_slow_norm
from sentinel.tools.orchestrate import enrich_prompt_with_tools
from sentinel.tools.catalog import list_mcp_tools
from sentinel.tools.weather import weather_report
from sentinel.tools.web_search import web_search_report
from sentinel.tools.timeutil import now_report
from sentinel.tools.calculator import calculate
from sentinel.tools.wikipedia import wikipedia_summary
from sentinel.tools.currency import convert_currency
from sentinel.serving.reply_polish import polish_reply
from sentinel.utils.logging import setup_production_logging
from sentinel.serving.production import load_profile, security_headers, is_production
from sentinel.utils.text_norm import normalize_prompt

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

_model: Optional[SENTINEL] = None
_cfg: Optional[SENTINELConfig] = None
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
    user_id: Optional[str] = Field(
        None, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$",
        description="Owner of the persistent slow memory + chat transcripts (needs NEUROFIELD_MEMORY_DIR).",
    )
    remember: bool = Field(True, description="False: do not store this turn or update the user's slow memory.")
    use_meta_rag: bool = Field(
        True,
        description="Metacognition-controlled Adaptive RAG: retrieve only when confidence is low / arousal high.",
    )
    force_rag: Optional[bool] = Field(
        None,
        description="Override meta decision: true=always retrieve, false=never, null=let metacognition decide.",
    )


# Sandbox request bodies live at MODULE level on purpose. With
# `from __future__ import annotations`, FastAPI resolves `req: ShellRequest` against the module
# globals; classes defined inside create_app() were unresolvable, so every call to
# /v1/tools/shell|python|reset used to fail with 422 ("query.req field required").
class ShellRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default-session", max_length=64)


class PythonRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=20_000)
    session_id: str = Field(default="default-session", max_length=64)


class ToolResetRequest(BaseModel):
    session_id: str = Field(default="default-session", max_length=64)


class ChatResponse(BaseModel):
    text: str
    session_id: str
    latency_ms: float
    model: str = ""
    meta_rag: Optional[dict] = Field(
        None,
        description="Adaptive RAG decision (should_retrieve, confidence, top_k, reason, hits).",
    )


# --------------------------------------------------------------------------
# Model loading (shared logic lives in sentinel.checkpoint.resolve)
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
        # Chat specials only when the checkpoint was trained with them.
        # Orphan specials in char vocab (never seen in data) make prompts like
        # <|user|>... and the model emits garbage — fall back to "User:"/"Assistant:".
        uses = meta.get("uses_specials")
        if uses is None:
            uses = bool(getattr(tok, "uses_specials", False)) and bool(_stop_token_ids)
            # If specials exist in vocab but meta never set the flag, prefer legacy
            # text roles for char tokenizers (User:/Assistant: training style).
            if getattr(tok, "uses_specials", False) and meta.get("tokenizer") == "char":
                uses = False
        set_use_specials(bool(uses))
        if not uses:
            _stop_token_ids = set()  # text heuristics for stop
        _model = model
        _cfg = full_cfg
        _sessions.clear()
        reset_history()  # no cross-test / cross-reload transcript leak
        _monitor = SafetyMonitor(
            z_threshold=full_cfg.safety.anomaly_z_threshold,
            log_path=None,
            freeze_after_alarms=int(getattr(full_cfg.safety, "freeze_after_alarms", 3) or 3),
            min_history=int(getattr(full_cfg.safety, "anomaly_min_history", 30) or 30),
        )
        _model_id = str(meta.get("model_id") or meta.get("name") or Path(checkpoint).stem or "model")
        _max_context = int(_env("MAX_CONTEXT", str(full_cfg.model.max_seq_len)) or full_cfg.model.max_seq_len)
        try:
            from sentinel.serving.chat_history import _max_turns
            logger.info(
                "Long-context limits | max_context_tokens=%s max_chat_turns=%s slow_slots=%s truncate_win=%s",
                _max_context,
                _max_turns(),
                getattr(full_cfg.model, "context_slots", 0) or "auto",
                getattr(full_cfg.model, "truncate_write_window", 32),
            )
        except Exception:
            pass
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

def _session_memory_frozen(sid: str) -> bool:
    """Paper §7: frozen sessions write with g=0."""
    with _lock:
        sess = _sessions.get(sid) or {}
        return bool(sess.get("memory_frozen") or sess.get("anomaly"))

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
    slow_seed: Optional[dict] = None,
    replay_len: int = 0,
) -> Iterator[tuple[int, Optional[dict]]]:
    """Yield (next_id, audit) while holding the model lock. Persists session state at the end."""
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    input_ids = input_ids[-_max_context:]
    with _lock:
        sess = _claim_session_locked(sid, owner)
        if reset:
            sess["state"] = None
            sess["memory_frozen"] = False
            if _monitor is not None:
                _monitor.reset_session(sid)
        sess["anomaly"] = False
        state = sess["state"]
        if state is None and slow_seed is not None:
            # fast memory/h start from zero; ONLY the user's slow ring is restored
            state = _model.new_state(1, _device, slow=slow_seed)
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
                if 0 < replay_len < cur.size(1):
                    # re-fed history: rebuilds the fast context but must NOT be written into the slow ring again
                    state = _model(cur[:, :replay_len], state=state, write_slow=False).state
                    cur = cur[:, replay_len:]
                for _ in range(max_new):
                    # Audit only when the client asked or the safety monitor is on —
                    # avoids extra collect_stats work on the hot path.
                    need_audit = bool(want_audit) or (_monitor is not None)
                    out = _model(cur, state=state, return_audit=need_audit, collect_stats=need_audit)
                    state = out.state
                    audit = out.audit
                    if _monitor is not None and audit:
                        alarms = _monitor.update(audit, session_id=sid)
                        if alarms:
                            # Flag the turn: slow ring must not be persisted for this user
                            sess["anomaly"] = True
                        if (
                            _cfg is not None
                            and _cfg.safety.freeze_on_anomaly
                            and _monitor.should_freeze(session_id=sid)
                        ):
                            # Paper §7: freeze memory writes → g=0 (state flag), not weight destroy
                            sess["memory_frozen"] = True
                            if state is not None:
                                state["memory_frozen"] = True
                            if alarms:
                                logger.warning(
                                    "Anomaly freeze (g=0) session=%s alarms=%s", sid, alarms
                                )
                    if state is not None and sess.get("memory_frozen"):
                        state["memory_frozen"] = True
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
    slow_seed: Optional[dict] = None,
    replay_len: int = 0,
) -> tuple[list[int], float, Optional[dict]]:
    start = time.perf_counter()
    new_ids: list[int] = []
    audit = None
    for nid, a in _token_stream(
        input_ids, max_new, temperature, sid, owner, reset, want_audit, slow_seed, replay_len
    ):
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
from sentinel.sandbox import SandboxConfig, SandboxRegistry

_sandbox_registry: SandboxRegistry | None = None
_memory: Optional[UserMemoryStore] = None  # per-user slow memory + transcripts (opt-in)


def _sandbox_id(sid: str, api_key: Optional[str]) -> str:
    """Per-owner sandbox key: two API keys can never address the same sandbox/audit log."""
    return hashlib.sha256(f"{api_key or ''}\0{sid}".encode()).hexdigest()[:32]


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
    product = _env("PRODUCT", "SENTINEL") or "SENTINEL"

    profile = load_profile()
    # Demo weights only in non-production (local dev convenience).
    if not checkpoint and profile.allow_demo_checkpoint:
        for candidate in (
            Path("checkpoints/examples/AERIS_main"),
            Path(__file__).resolve().parents[3] / "checkpoints" / "examples" / "AERIS_main",
            Path.cwd() / "checkpoints" / "examples" / "AERIS_main",
        ):
            if (candidate / "model.safetensors").is_file():
                checkpoint = str(candidate)
                logger.info("No NEUROFIELD_CHECKPOINT set — using demo weights at %s", checkpoint)
                break
    if not checkpoint:
        logger.warning(
            "No checkpoint configured. Set NEUROFIELD_CHECKPOINT=/path/to/folder "
            "(folder must contain model.safetensors) or chat will return 503 model not loaded."
        )
    if profile.production:
        keys_ok = bool(get_api_keys())
        issues = profile.issues(has_api_keys=keys_ok, checkpoint_set=bool(checkpoint))
        for msg in issues:
            logger.error("PRODUCTION GATE: %s", msg)
        if issues:
            logger.error(
                "Refusing soft-start: fix env before exposing this process. "
                "Run: PYTHONPATH=src python scripts/ops/production_check.py"
            )

    _prof = load_profile()
    app = FastAPI(
        title=f"{product} API",
        description=f"{product} — SENTINEL two-speed memory architecture.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if _prof.expose_docs else None,
        redoc_url="/redoc" if _prof.expose_docs else None,
        openapi_url="/openapi.json" if _prof.expose_docs else None,
    )
    app.state.production_profile = _prof

    _sys = _env("SYSTEM_PROMPT") or default_system_prompt()
    set_system_prompt(_sys)
    logger.info("System prompt set (%d chars)", len(_sys or ""))

    # Backend = API only. Canonical UI is apps/frontend (Bun :3000).
    # SERVE_UI=1 may serve THAT same frontend folder for local debug — No HTML under serving/.
    _repo_root = Path(__file__).resolve().parents[3]
    frontend_public = _repo_root / "apps" / "frontend" / "public"
    static_dir = frontend_public if frontend_public.is_dir() else (Path(__file__).parent / "static")
    serve_ui = _env("SERVE_UI", "0") == "1"
    if serve_ui and static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info("SERVE_UI=1 — serving frontend from %s (prefer Bun :3000 in production)", static_dir)

    global _memory
    _memory = None
    mem_dir = _env("MEMORY_DIR")
    if mem_dir:
        if get_api_keys() or _env("MEMORY_ALLOW_OPEN", "0") == "1":
            _memory = UserMemoryStore(mem_dir, max_chat_mb=float(_env("MEMORY_MAX_CHAT_MB", "10") or 10))
        else:
            logger.warning(
                "NEUROFIELD_MEMORY_DIR is set but NEUROFIELD_API_KEYS is empty: persistent memory stays OFF "
                "(transcripts are personal data). Set API keys, or NEUROFIELD_MEMORY_ALLOW_OPEN=1 for local dev."
            )

    if checkpoint:
        try:
            load_model(checkpoint, config_path, device)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to load checkpoint %s: %s", checkpoint, e)
            # leave _model None — /v1/chat will still return a clear 503
    else:
        logger.error("Model not loaded: no checkpoint path available")

    def _owner(api_key: Optional[str]) -> Optional[str]:
        # Session ownership is enforced only when auth is enabled.
        return api_key

    def _limit(request: Request, api_key: Optional[str]) -> None:
        _limiter.check(api_key or client_ip(request))

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        for k, v in security_headers().items():
            response.headers.setdefault(k, v)
        # request id for support / abuse traces
        rid = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        if rid:
            response.headers.setdefault("X-Request-ID", rid[:64])
        return response

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """Apply rate limit to all /v1/* API routes (product, tools, chat, agent)."""
        path = request.url.path or ""
        if path.startswith("/v1/"):
            try:
                # Prefer API key bucket when present
                key = request.headers.get("x-api-key") or client_ip(request)
                _limiter.check(key)
            except HTTPException as e:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=e.status_code,
                    content={"detail": e.detail},
                    headers=dict(e.headers or {}),
                )
        return await call_next(request)

    def _need_model() -> None:
        if _model is None or _encode is None or _decode is None:
            raise HTTPException(status_code=503, detail="model not loaded")

    async def _metrics_guard(request: Request) -> None:
        if _env("METRICS_PUBLIC", "1") == "1":
            return
        await require_api_key(request.headers.get("x-api-key"))

    @app.get("/")
    async def root():
        if serve_ui:
            index = static_dir / "index.html"
            if index.is_file():
                return FileResponse(str(index))
        # Pure backend response — no HTML
        return {
            "service": "aeris-python-backend",
            "role": "model_inference",
            "stack": "python",
            "product": product,
            "version": __version__,
            "model_loaded": _model is not None,
            "model": _model_id or None,
            "endpoints": {
                "docs": "/docs",
                "healthz": "/healthz",
                "readyz": "/readyz",
                "chat": "POST /v1/chat",
                "completions": "POST /v1/completions",
                "models": "GET /v1/models",
            },
            "frontend": "http://localhost:3000",
            "production": is_production(),
            "limits": {
                "policy": "Megha+Bhutu block unauthorized cyber activity",
                "model": "Quality depends on trained checkpoint; not a general frontier LM guarantee",
            },
        }

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "product": product, "version": __version__}

    @app.get("/readyz")
    async def readyz():
        prof = load_profile()
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "model not loaded — set NEUROFIELD_CHECKPOINT to a folder that contains "
                    "model.safetensors, then restart uvicorn"
                ),
            )
        if prof.production and not get_api_keys():
            raise HTTPException(
                status_code=503,
                detail="production requires NEUROFIELD_API_KEYS",
            )
        return {
            "status": "ready",
            "device": str(_device),
            "model": _model_id,
            "checkpoint": _checkpoint_path,
            "has_tokenizer": _encode is not None,
            "params": _model.param_report()["total"],
            "production": prof.production,
            "bhutu": True,
            "megha_policy": True,
        }

    @app.get("/metrics", dependencies=[Depends(_metrics_guard)])
    async def metrics():
        with _stats_lock:
            req, err, lat = _stats["requests"], _stats["errors"], _stats["latency_ms"]
        body = (
            "# HELP sentinel_requests_total Total requests\n"
            "# TYPE sentinel_requests_total counter\n"
            f"sentinel_requests_total {int(req)}\n"
            "# HELP sentinel_errors_total Total errors\n"
            "# TYPE sentinel_errors_total counter\n"
            f"sentinel_errors_total {int(err)}\n"
            "# HELP sentinel_latency_ms Average latency\n"
            "# TYPE sentinel_latency_ms gauge\n"
            f"sentinel_latency_ms {lat / max(1, req):.2f}\n"
            "# HELP sentinel_sessions Active sessions\n"
            "# TYPE sentinel_sessions gauge\n"
            f"sentinel_sessions {len(_sessions)}\n"
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
                    "owned_by": "sentinel",
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

    def _uid(req: ChatRequest) -> str:
        return req.user_id or "default"

    def _prepare_chat(
        req: ChatRequest, sid: str, owner: Optional[str] = None
    ) -> tuple[str, list[int], int, float, int, Optional[dict], Optional[dict]]:
        prompt = normalize_prompt(req.prompt)
        slow_seed = None
        if _memory is not None and req.remember:
            uid = _uid(req)
            if history_len(sid) == 0:  # new process / TTL eviction: the chat window is NOT lost
                restore_turns(sid, _memory.recent_turns(owner, uid, sid, n=12))
            slow_seed = _memory.load_slow(owner, uid)

        # --- Auto RAG (metacognition-controlled, mode=auto by default) ---
        # Retrieves for informational prompts; skips pure chit-chat.
        # Live surprise from prior tokens feeds confidence when available.
        meta_rag_info: Optional[dict] = None
        rag_prompt = prompt
        if req.use_meta_rag or req.force_rag is True:
            try:
                from sentinel.rag.meta_rag import auto_rag

                conf = arou = confict = None
                with _lock:
                    sess = _sessions.get(sid) or {}
                    state = sess.get("state") or {}
                    last_S = state.get("last_S")
                    if last_S is not None:
                        try:
                            s_val = float(last_S.mean().item()) if hasattr(last_S, "mean") else float(last_S)
                            conf = max(0.05, min(0.95, 1.0 / (1.0 + s_val)))
                            arou = max(0.0, s_val)
                        except Exception:
                            pass
                rag_prompt, meta_rag_info = auto_rag(
                    prompt,
                    confidence=conf,
                    arousal=arou,
                    conflict=confict,
                    force=req.force_rag,
                    session_key=sid or "default",
                )
            except Exception as e:
                logger.warning("auto_rag failed, continuing without retrieval: %s", e)
                rag_prompt = prompt

        # Real-world facts (weather / web) injected when tools enabled
        enriched, _tool_note = enrich_prompt_with_tools(rag_prompt)
        input_ids = _encode(build_prompt(sid, enriched)) or _encode(enriched) or [0]
        n_user = len(_encode(build_user_part(enriched)) or [])
        replay_len = max(0, len(input_ids) - n_user) if slow_seed is not None else 0
        max_new, temperature = adaptive_sampling(
            prompt,
            user_max=req.max_new_tokens,
            user_temp=req.temperature,
            auto=req.auto_length,
        )
        return prompt, input_ids, max_new, temperature, replay_len, slow_seed, meta_rag_info

    def _persist(sid: str, owner: Optional[str], req: ChatRequest, prompt: str, text: str) -> None:
        """Lossless transcript + the user's slow ring (after norm cap + audit), never from a flagged turn."""
        if _memory is None or not req.remember:
            return
        uid = _uid(req)
        _memory.append_turn(owner, uid, sid, "user", prompt)
        _memory.append_turn(owner, uid, sid, "assistant", text)
        with _lock:
            sess = _sessions.get(sid)
            state = sess["state"] if sess else None
            anomaly = bool(sess and sess.get("anomaly"))
        if _model is not None and state is not None and not anomaly:
            state = _model.consolidate_turn(state)  # turn-end snapshot of the fast memory into the user's ring
        slow = _model.export_slow_state(state) if (_model is not None and state is not None) else None
        if slow is None:
            return
        if anomaly:
            _memory.append_audit(owner, uid, {"event": "slow_memory_not_saved", "reason": "anomaly flagged", "session": sid})
            return
        cap = float(getattr(getattr(_cfg, "safety", None), "max_memory_norm", 0.0) or 0.0)
        slow, capped, biggest = cap_slow_norm(slow, cap)
        _memory.save_slow(owner, uid, slow)
        _memory.append_audit(owner, uid, {
            "event": "slow_memory_saved", "session": sid,
            "slots_used": int((slow["slow_filled"] > 0).sum()), "writes_represented": float(slow["slow_filled"].sum()),
            "largest_slot_norm": round(biggest, 4), "slots_capped": capped,
        })

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(
        req: ChatRequest,
        request: Request,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        _limit(request, api_key)
        _bump("requests")
        sid = _valid_session_id(req.session_id)
        try:
            # Buddy companion commands (decorative only — no tools/model)
            _raw0 = normalize_prompt(req.prompt)
            try:
                from sentinel.buddy import is_buddy_command, handle_buddy_command
                if is_buddy_command(_raw0):
                    owner = _owner(api_key) or (api_key or sid)
                    text = handle_buddy_command(_raw0, str(owner))
                    if text:
                        # Control tokens for UI; short human message for transcript
                        if text == "BUDDY_ON":
                            msg = "Buddy on."
                            meta = {"buddy": True, "buddy_ui": "on"}
                        elif text == "BUDDY_OFF":
                            msg = "Buddy off."
                            meta = {"buddy": True, "buddy_ui": "off"}
                        else:
                            msg = text
                            meta = {"buddy": True}
                        append_turn(sid, "user", _raw0)
                        append_turn(sid, "assistant", msg)
                        return ChatResponse(
                            text=msg,
                            session_id=sid,
                            latency_ms=0.0,
                            model="buddy",
                            meta_rag=meta,
                        )
            except Exception as e:
                logger.warning("buddy command: %s", e)

            # Bhutu (Metacognition manager) + Megha policy — full tool/agent control gate
            try:
                from sentinel.modules.bhutu import bhutu_decide
                _b0 = bhutu_decide(_raw0)
                if not _b0.allow:
                    msg = _b0.user_message or "Request blocked by Bhutu."
                    append_turn(sid, "user", _raw0)
                    append_turn(sid, "assistant", msg)
                    return ChatResponse(
                        text=msg,
                        session_id=sid,
                        latency_ms=0.0,
                        model="bhutu-policy",
                        meta_rag={"bhutu": _b0.to_dict()},
                    )
                # stash for SI_v1 path (same process)
                request.state.bhutu = _b0
            except Exception as e:
                logger.warning("bhutu precheck: %s", e)

            # --- Tool-first path (Grok-style live facts) ---
            # Time / calc / weather / search / wiki answer from tools, not weak LM decode.
            # Disable: NEUROFIELD_CHAT_TOOLS_DIRECT=0
            direct = (_env("CHAT_TOOLS_DIRECT", "1") or "1").strip().lower() not in {
                "0", "false", "no", "off",
            }
            if direct:
                t0 = time.perf_counter()
                raw = normalize_prompt(req.prompt)
                # Metacognition + Megha policy monitor: refuse illegal/unauthorized cyber activity
                try:
                    from sentinel.megha.policy import evaluate_user_text
                    _pol = evaluate_user_text(raw)
                    if not _pol.allow:
                        lat = (time.perf_counter() - t0) * 1000.0
                        append_turn(sid, "user", raw)
                        msg = _pol.user_message or "Request blocked by Megha security policy."
                        append_turn(sid, "assistant", msg)
                        return ChatResponse(
                            text=msg,
                            session_id=sid,
                            latency_ms=lat,
                            model="megha-policy",
                            meta_rag={"policy": _pol.to_dict()},
                        )
                    if _pol.action == "route_megha" and _pol.megha_mode:
                        from sentinel.megha import MeghaRequest, run_megha
                        _mr = run_megha(
                            MeghaRequest(
                                mode=_pol.megha_mode,
                                text=raw,
                                authorized=False,
                            )
                        )
                        if _mr.ok and _mr.items:
                            lat = (time.perf_counter() - t0) * 1000.0
                            lines = [ _mr.summary or "Megha", "" ]
                            for it in _mr.items[:20]:
                                if isinstance(it, dict):
                                    lines.append("- " + "; ".join(f"{k}: {v}" for k, v in it.items() if v is not None))
                                else:
                                    lines.append(f"- {it}")
                            if _mr.warnings:
                                lines.append("")
                                lines.extend(f"Note: {w}" for w in _mr.warnings[:3])
                            text = "\n".join(lines)
                            append_turn(sid, "user", raw)
                            append_turn(sid, "assistant", text)
                            return ChatResponse(
                                text=text,
                                session_id=sid,
                                latency_ms=lat,
                                model="megha+meta-control",
                                meta_rag={"policy": _pol.to_dict(), "megha": _mr.to_dict()},
                            )
                except Exception as e:
                    logger.warning("Megha policy monitor failed open to tools/model: %s", e)
                # Prefer SI_v1 plan→tool→verify (Grok-style agent loop)
                text = None
                try:
                    from sentinel.agent.si_v1 import run_si_v1, SI_v1Config, build_plan
                    plan = build_plan(raw)
                    # Only short-circuit when plan actually uses a tool (not compose-only)
                    uses_tool = any(
                        s.get("action") not in ("compose_answer", "verify_nonempty", "")
                        for s in plan
                    )
                    if uses_tool:
                        result = run_si_v1(raw, config=SI_v1Config(max_steps=6))
                        if result.answer and result.stopped_reason not in (
                            "empty_input",
                            "no_tools",
                        ):
                            text = result.answer.strip()
                except Exception as e:
                    logger.warning("SI_v1 chat path failed, falling back to enrich: %s", e)
                if not text:
                    _enriched, tool_note = enrich_prompt_with_tools(raw)
                    if tool_note and str(tool_note).strip():
                        text = str(tool_note).strip()
                if text:
                    lat = (time.perf_counter() - t0) * 1000.0
                    append_turn(sid, "user", raw)
                    append_turn(sid, "assistant", text)
                    _persist(sid, _owner(api_key), req, raw, text)
                    return ChatResponse(
                        text=text,
                        session_id=sid,
                        latency_ms=lat,
                        model="tools+si_v1",
                        meta_rag=None,
                    )

            _need_model()
            prompt, input_ids, max_new, temperature, replay_len, slow_seed, meta_rag_info = _prepare_chat(
                req, sid, _owner(api_key)
            )
            # History is re-fed as text every turn, so start from a fresh FAST state (reset=True);
            # only the user's persistent SLOW ring (if any) is restored, and the re-fed history is
            # replayed without writing the ring again (replay_len).
            new_ids, lat, _ = _generate(
                input_ids, max_new, temperature, sid, _owner(api_key), reset=True,
                slow_seed=slow_seed, replay_len=replay_len,
            )
            text = _postprocess(prompt, _decode(new_ids), req.postprocess)
            # Optional: model emitted JSON tool call → execute (Grok-style)
            _mtc = (_env("MODEL_TOOL_CALLS", "1") or "1").strip().lower() not in {
                "0", "false", "no", "off",
            }
            if _mtc and text:
                try:
                    from sentinel.serving.tool_calls import extract_tool_call, run_tool_call
                    call = extract_tool_call(text)
                    if call:
                        tool_out = run_tool_call(call)
                        text = tool_out.strip() or text
                        logger.info("model tool call executed: %s", call.get("tool"))
                except Exception as e:
                    logger.warning("model tool call failed: %s", e)
            append_turn(sid, "user", prompt)
            append_turn(sid, "assistant", text)
            _persist(sid, _owner(api_key), req, prompt, text)
            return ChatResponse(
                text=text, session_id=sid, latency_ms=lat, model=_model_id, meta_rag=meta_rag_info
            )
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
        owner = _owner(api_key)
        prompt, input_ids, max_new, temperature, replay_len, slow_seed, meta_rag_info = _prepare_chat(
            req, sid, owner
        )

        def event_gen() -> Iterator[str]:
            start = time.perf_counter()
            new_ids: list[int] = []
            start_payload = {"session_id": sid, "event": "start"}
            if meta_rag_info is not None:
                start_payload["meta_rag"] = meta_rag_info
            yield f"data: {json.dumps(start_payload)}\n\n"
            try:
                for nid, _ in _token_stream(
                    input_ids, max_new, temperature, sid, owner, reset=True,
                    slow_seed=slow_seed, replay_len=replay_len,
                ):
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
            _persist(sid, owner, req, prompt, text)
            done_payload = {"event": "done", "latency_ms": lat, "text": text}
            if meta_rag_info is not None:
                done_payload["meta_rag"] = meta_rag_info
            yield f"data: {json.dumps(done_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Per-user persistent memory (opt-in: NEUROFIELD_MEMORY_DIR). Always needs an API key.
    # ------------------------------------------------------------------
    def _need_memory() -> UserMemoryStore:
        if _memory is None:
            raise HTTPException(status_code=404, detail="persistent memory is disabled (set NEUROFIELD_MEMORY_DIR and API keys)")
        return _memory

    @app.get("/v1/memory")
    def memory_info(user_id: str = "default", api_key: str = Depends(require_api_key_strict)):
        return _need_memory().info(api_key or None, user_id)

    @app.get("/v1/memory/transcript")
    def memory_transcript(
        session_id: str, user_id: str = "default", limit: int = 100,
        api_key: str = Depends(require_api_key_strict),
    ):
        if not _SESSION_ID_RE.match(session_id):
            raise HTTPException(status_code=422, detail="invalid session_id")
        turns = _need_memory().transcript(api_key or None, user_id, session_id, limit=max(1, min(limit, 1000)))
        return {"user_id": user_id, "session_id": session_id, "turns": turns}

    @app.delete("/v1/memory")
    def memory_delete(user_id: str = "default", api_key: str = Depends(require_api_key_strict)):
        existed = _need_memory().delete_user(api_key or None, user_id)
        reset_history()  # drop in-process windows too: nothing of the user may linger
        return {"status": "deleted" if existed else "nothing stored", "user_id": user_id}

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
            _sandbox().drop(_sandbox_id(session_id, api_key))
        return {"status": "reset", "session_id": session_id}

    # ------------------------------------------------------------------
    # Sandbox tools (production) — disabled when NEUROFIELD_SANDBOX=0
    # ------------------------------------------------------------------
    if __import__("os").environ.get("NEUROFIELD_SANDBOX", "1") != "0":
        # Code execution: API key REQUIRED (503 when none configured), rate-limited,
        # and every sandbox/audit log is scoped to the calling key.
        @app.post("/v1/tools/shell")
        def tools_shell(
            req: ShellRequest, request: Request, api_key: str = Depends(require_api_key_strict)
        ):
            _limit(request, api_key)
            sid = _valid_session_id(req.session_id)
            result = _sandbox().get(_sandbox_id(sid, api_key)).run_shell(req.command)
            return result.to_dict()

        @app.post("/v1/tools/python")
        def tools_python(
            req: PythonRequest, request: Request, api_key: str = Depends(require_api_key_strict)
        ):
            _limit(request, api_key)
            sid = _valid_session_id(req.session_id)
            result = _sandbox().get(_sandbox_id(sid, api_key)).run_python(req.code)
            return result.to_dict()

        @app.get("/v1/tools/audit")
        def tools_audit(
            request: Request,
            session_id: str = "default-session",
            api_key: str = Depends(require_api_key_strict),
        ):
            _limit(request, api_key)
            sid = _valid_session_id(session_id)
            sb = _sandbox().get(_sandbox_id(sid, api_key))
            return {"session_id": sid, "calls": sb.calls, "audit": sb.audit()}

        @app.post("/v1/tools/reset")
        def tools_reset(
            req: ToolResetRequest, request: Request, api_key: str = Depends(require_api_key_strict)
        ):
            _limit(request, api_key)
            sid = _valid_session_id(req.session_id)
            _sandbox().reset(_sandbox_id(sid, api_key))
            return {"status": "reset", "session_id": sid}


    # ------------------------------------------------------------------
    # Web / weather tools + MCP-style catalog (NEUROFIELD_WEB_TOOLS=0 to disable chat auto-use)
    # ------------------------------------------------------------------
    @app.get("/v1/tools/mcp")
    def tools_mcp_list(api_key: Optional[str] = Depends(require_api_key)):
        """MCP-style tools/list payload for clients and agents."""
        return list_mcp_tools()

    @app.post("/v1/rag/retrieve")
    def rag_retrieve_api(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        """RAG: retrieve + rank chunks from the domain knowledge base."""
        from sentinel.rag.pipeline import get_default_rag

        q = str((body or {}).get("query") or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="query required")
        try:
            k = max(1, min(20, int((body or {}).get("top_k") or 4)))
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="top_k must be an integer") from e
        rag = get_default_rag()
        hits = rag.retrieve(q, top_k=k)
        return {"query": q[:500], "hits": hits, "store_size": len(rag.store)}

    @app.post("/v1/rag/ingest")
    def rag_ingest_api(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        """RAG: add text into the vector store (continuous update)."""
        from sentinel.rag.pipeline import get_default_rag

        text = str((body or {}).get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        source = str((body or {}).get("source") or "api")[:120]
        rag = get_default_rag()
        n = rag.ingest_text(text, source=source)
        return {"ingested": n, "store_size": len(rag.store), "source": source}

    @app.post("/v1/tools/web_search")
    def tools_web_search(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        q = str((body or {}).get("query") or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="query required")
        try:
            n = max(1, min(10, int((body or {}).get("max_results") or 5)))
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="max_results must be an integer") from e
        return {"query": q[:300], "report": web_search_report(q[:300], max_results=n)}

    @app.post("/v1/tools/weather")
    def tools_weather(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        loc = str((body or {}).get("location") or "").strip()
        if not loc:
            raise HTTPException(status_code=400, detail="location required")
        return {"location": loc, "report": weather_report(loc)}

    @app.post("/v1/tools/time")
    def tools_time(body: dict | None = None, api_key: Optional[str] = Depends(require_api_key)):
        body = body or {}
        tz = str(body.get("timezone") or "").strip() or None
        return {"report": now_report(tz)}

    @app.post("/v1/tools/calculator")
    def tools_calculator(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        expr = str((body or {}).get("expression") or "").strip()
        if not expr:
            raise HTTPException(status_code=400, detail="expression required")
        return {"report": calculate(expr)}

    @app.post("/v1/tools/wikipedia")
    def tools_wikipedia(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        topic = str((body or {}).get("topic") or "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="topic required")
        lang = str((body or {}).get("lang") or "en").strip() or "en"
        return {"report": wikipedia_summary(topic, lang=lang)}

    @app.post("/v1/tools/currency")
    def tools_currency(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        try:
            amount = float((body or {}).get("amount"))
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="amount required") from e
        fr = str((body or {}).get("from") or "").strip()
        to = str((body or {}).get("to") or "").strip()
        if not fr or not to:
            raise HTTPException(status_code=400, detail="from and to required")
        return {"report": convert_currency(amount, fr, to)}

    @app.post("/v1/agent")
    def agent_run(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        """Reliable agent: tool enrich + optional model polish. Hard step budget."""
        from sentinel.agent import AgentConfig, run_agent

        prompt = str((body or {}).get("prompt") or (body or {}).get("question") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt required")
        use_model = bool((body or {}).get("use_model", False))
        max_steps = int((body or {}).get("max_steps") or 4)

        generate_fn = None
        if use_model:
            # reuse chat path lightly: one completion from global model if loaded
            def generate_fn(enriched: str) -> str:
                # Prefer existing generate helper if present on app state
                gen = getattr(app.state, "generate_text", None)
                if callable(gen):
                    return gen(enriched)
                # Fallback: empty -> facts-only agent behaviour
                return ""

        result = run_agent(
            prompt,
            generate_fn=generate_fn if use_model else None,
            config=AgentConfig(max_steps=max_steps),
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Product UX: Kalofacts library + AjeebWork plan (optional layer)
    # ------------------------------------------------------------------
    @app.get("/product")
    def product_page():
        """Product shell HTML — only when SERVE_UI=1; canonical host is Bun :3000/product."""
        if not serve_ui:
            raise HTTPException(
                status_code=404,
                detail="Product UI is on the frontend (:3000/product). Backend is API-only (set SERVE_UI=1 only for local debug).",
            )
        path = static_dir / "product.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="product.html missing under apps/frontend/public")
        return FileResponse(str(path))

    @app.get("/v1/kalofacts")
    def kalofacts_list(api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.kalofacts.store import list_artifacts
        return {"artifacts": list_artifacts()}

    @app.post("/v1/kalofacts")
    def kalofacts_create(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.kalofacts.store import create_and_save
        title = str((body or {}).get("title") or "Untitled").strip() or "Untitled"
        content = str((body or {}).get("content") or "")
        kind = str((body or {}).get("kind") or "text")
        art = create_and_save(title=title, content=content, kind=kind)
        return art.to_dict() if hasattr(art, "to_dict") else art

    @app.get("/v1/kalofacts/{artifact_id}")
    def kalofacts_get(artifact_id: str, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.kalofacts.store import get
        art = get(artifact_id)
        if art is None:
            raise HTTPException(status_code=404, detail="not found")
        return art.to_dict() if hasattr(art, "to_dict") else art

    @app.put("/v1/kalofacts/{artifact_id}")
    def kalofacts_update(artifact_id: str, body: dict, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.kalofacts.store import get, save
        from sentinel.kalofacts.edit import replace_all_content
        art = get(artifact_id)
        if art is None:
            raise HTTPException(status_code=404, detail="not found")
        content = (body or {}).get("content")
        if content is not None:
            art = replace_all_content(art, str(content))
        title = (body or {}).get("title")
        if title is not None:
            art.title = str(title)
        save(art)
        return art.to_dict() if hasattr(art, "to_dict") else art

    @app.delete("/v1/kalofacts/{artifact_id}")
    def kalofacts_delete(artifact_id: str, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.kalofacts.store import delete
        ok = delete(artifact_id)
        if not ok:
            raise HTTPException(status_code=404, detail="not found")
        return {"status": "deleted", "id": artifact_id}

    @app.post("/v1/ajeebwork/plan")
    def ajeeb_plan(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.ajeebwork.plan_execute import build_plan, plan_execute_report
        objective = str((body or {}).get("objective") or "").strip()
        if not objective:
            raise HTTPException(status_code=400, detail="objective required")
        step_specs = (body or {}).get("steps") or []
        if not isinstance(step_specs, list):
            step_specs = []
        # default one noop step if empty — structure only
        if not step_specs:
            step_specs = [{"title": "Review objective", "action": "note", "args": {"text": objective}}]
        report = plan_execute_report(objective, step_specs, run_step=None)
        steps = build_plan(objective, step_specs)
        return {
            "objective": objective,
            "report": report,
            "steps": [s.__dict__ if hasattr(s, "__dict__") else s for s in steps],
        }

    @app.post("/v1/si/v1")
    def si_v1_endpoint(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.agent.si_v1 import run_si_v1, SI_v1Config
        prompt = str((body or {}).get("prompt") or (body or {}).get("question") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt required")
        max_steps = int((body or {}).get("max_steps") or 6)
        return run_si_v1(prompt, config=SI_v1Config(max_steps=max_steps)).to_dict()

    # ---- Megha: defensive cybersecurity / authorized testing assistant ----
    @app.get("/v1/buddy")
    def buddy_get(
        api_key: Optional[str] = Depends(require_api_key),
        user_id: Optional[str] = None,
    ):
        from sentinel.buddy import hatch_buddy
        owner = (user_id or "").strip() or (_owner(api_key) if api_key else "anonymous")
        b = hatch_buddy(str(owner or "anonymous"))
        return {
            "product": "Buddy",
            "buddy": b.to_public_dict(),
            "ascii": b.render(),
            "placement": "float_below_navbar",
        }

    @app.post("/v1/buddy")
    def buddy_post(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.buddy import hatch_buddy, pet_buddy, handle_buddy_command
        body = body or {}
        # Prefer explicit user_id from UI so hatch stays stable per browser
        owner = str(body.get("user_id") or "").strip() or (
            _owner(api_key) if api_key else "anonymous"
        )
        cmd = str(body.get("command") or "/buddy").strip()
        b = hatch_buddy(str(owner))
        if "pet" in cmd.lower():
            return {
                "product": "Buddy",
                "text": pet_buddy(str(owner)),
                "buddy": b.to_public_dict(),
                "placement": "float_below_navbar",
            }
        text = handle_buddy_command(cmd if cmd.startswith("/") else "/buddy", str(owner))
        return {
            "product": "Buddy",
            "text": text,
            "buddy": b.to_public_dict(),
            "placement": "float_below_navbar",
        }

    @app.get("/v1/megha/capabilities")
    def megha_capabilities(api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.megha import list_capabilities
        return {"product": "Megha", "capabilities": list_capabilities()}

    @app.post("/v1/megha")
    def megha_run(body: dict, api_key: Optional[str] = Depends(require_api_key)):
        from sentinel.megha import MeghaRequest, run_megha
        b = body or {}
        req = MeghaRequest(
            mode=str(b.get("mode") or "help"),
            text=str(b.get("text") or b.get("source") or b.get("symptom") or ""),
            target_type=str(b.get("target_type") or b.get("focus") or "web"),
            framework=str(b.get("framework") or "general"),
            authorized=bool(b.get("authorized") or False),
            scope=str(b.get("scope") or ""),
            attestor=str(b.get("attestor") or ""),
        )
        result = run_megha(req)
        status = 200 if result.ok else (403 if result.error and "authorized" in (result.error or "").lower() else 400)
        if status != 200:
            raise HTTPException(status_code=status, detail=result.to_dict())
        return result.to_dict()

    return app


# uvicorn entry: sentinel.serving.server:create_app --factory
