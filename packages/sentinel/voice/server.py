"""Indo-Voice-3 production WebSocket / HTTP service."""

from __future__ import annotations

import base64
import logging
import uuid
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sentinel.voice_settings import get_settings
from sentinel.voice.factory import build_voice_system
from sentinel.voice.stt_tts import AudioChunk

logger = logging.getLogger("indo_voice.server")


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    app = FastAPI(title="Indo-Voice-3", version="1.0.0", docs_url="/docs")
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.ready = False
    app.state.adapter = None
    app.state.stt = None
    app.state.tts = None
    app.state.tokenizer = None
    app.state.sessions: Dict[str, object] = {}

    @app.on_event("startup")
    async def startup():
        try:
            adapter, model, tok, stt, tts, _ = build_voice_system(settings, require_checkpoint=True)
            app.state.adapter = adapter
            app.state.stt = stt
            app.state.tts = tts
            app.state.tokenizer = tok
            app.state.ready = True
            logger.info("Indo-Voice-3 ready device=%s stt=%s tts=%s", settings.device, type(stt).__name__, type(tts).__name__)
        except Exception:
            logger.exception("Startup failed — check INDO_CHECKPOINT and dependencies")
            app.state.ready = False

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if app.state.ready else "degraded",
            "ready": app.state.ready,
            "device": settings.device,
            "checkpoint": settings.checkpoint_dir,
            "stt": type(app.state.stt).__name__ if app.state.stt else None,
            "tts": type(app.state.tts).__name__ if app.state.tts else None,
        }

    @app.post("/v1/session")
    async def create_session(language: Optional[str] = None):
        if not app.state.ready:
            raise HTTPException(503, "Service not ready")
        lang = language or settings.primary_language
        sid = str(uuid.uuid4())
        session = app.state.adapter.create_session(sid, language=lang)
        app.state.sessions[sid] = session
        return {"session_id": sid, "language": lang}

    @app.websocket("/v1/voice")
    async def voice_socket(websocket: WebSocket):
        await websocket.accept()
        if not app.state.ready:
            await websocket.send_json({"type": "error", "code": "not_ready", "message": "Model not loaded"})
            await websocket.close(code=1013)
            return

        session = None
        session_id = None
        try:
            import json
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") != "session.init":
                await websocket.send_json({"type": "error", "message": "First event must be session.init"})
                await websocket.close()
                return

            session_id = msg.get("session_id") or str(uuid.uuid4())
            language = msg.get("language") or settings.primary_language
            session = app.state.sessions.get(session_id)
            if session is None:
                session = app.state.adapter.create_session(session_id, language=language)
                app.state.sessions[session_id] = session

            await websocket.send_json({"type": "session.ready", "session_id": session_id, "language": language})

            while True:
                data = await websocket.receive()
                if "text" in data:
                    await _on_text(websocket, json.loads(data["text"]), session, app)
                elif "bytes" in data:
                    await _on_audio(websocket, data["bytes"], session, app)

        except WebSocketDisconnect:
            logger.info("disconnect session=%s", session_id)
        except Exception:
            logger.exception("websocket error session=%s", session_id)
            try:
                await websocket.send_json({"type": "error", "message": "internal_error"})
            except Exception:
                pass

    return app


async def _on_text(ws, msg: dict, session, app):
    if msg.get("type") != "text.input":
        return
    text = (msg.get("text") or "").strip()
    if not text:
        return
    await _generate_and_speak(ws, text, session, app)


async def _on_audio(ws, raw: bytes, session, app):
    settings = app.state.settings
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return
    chunk = AudioChunk(
        samples=samples,
        sample_rate=settings.sample_rate,
        duration_sec=len(samples) / float(settings.sample_rate),
    )
    transcript = app.state.stt.transcribe(chunk, language=settings.primary_language)
    await ws.send_json({"type": "transcript", "text": transcript})
    if transcript.strip():
        await _generate_and_speak(ws, transcript, session, app)


async def _generate_and_speak(ws, user_text: str, session, app):
    adapter = app.state.adapter
    tok = app.state.tokenizer
    settings = app.state.settings
    if tok is None:
        await ws.send_json({"type": "error", "message": "tokenizer_missing"})
        return

    input_ids = tok.encode(user_text, return_tensors="pt")
    if input_ids.device.type != settings.device:
        input_ids = input_ids.to(settings.device)

    await ws.send_json({"type": "response.start"})
    pieces = []
    for token_id in adapter.generate_stream(
        input_ids,
        session,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    ):
        piece = tok.decode([token_id])
        pieces.append(piece)
        await ws.send_json({"type": "response.token", "token": piece, "token_id": int(token_id)})

    full = "".join(pieces).strip()
    await ws.send_json({"type": "response.text_done", "text": full})

    if full:
        try:
            audio = app.state.tts.synthesize(full, language=settings.primary_language)
            pcm = (np.clip(audio.samples, -1, 1) * 32767).astype(np.int16).tobytes()
            await ws.send_json({
                "type": "response.audio",
                "format": "pcm16",
                "sample_rate": audio.sample_rate,
                "data": base64.b64encode(pcm).decode("ascii"),
            })
        except Exception as e:
            logger.warning("TTS failed: %s", e)
            await ws.send_json({"type": "response.audio_error", "message": str(e)})

    await ws.send_json({"type": "response.done"})
