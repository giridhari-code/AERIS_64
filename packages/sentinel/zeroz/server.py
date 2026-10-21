"""0z code HTTP API — company-standard agent service."""

from __future__ import annotations

import os
from typing import Any, Optional

from sentinel.zeroz.config import ZeroZConfig, load_zeroz_config
from sentinel.zeroz.policy import PRODUCT_NAME, PRODUCT_TAGLINE, SYSTEM_PROMPT
from sentinel.zeroz.runtime import build_generate_fn_from_checkpoint, run_zeroz
from sentinel.zeroz.session import ZeroZSession
from sentinel.zeroz.toolbus import ToolBus

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install fastapi uvicorn pydantic  OR  pip install -e '.[voice]'") from e


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    session_id: Optional[str] = None
    use_model: bool = True


class ChatResponse(BaseModel):
    product: str = PRODUCT_NAME
    answer: str
    stopped_reason: str
    steps: int
    metrics: dict[str, Any]
    session_id: str
    tool_events: list[dict[str, Any]]


def create_app(cfg: ZeroZConfig | None = None) -> FastAPI:
    cfg = cfg or load_zeroz_config(os.getenv("ZEROZ_CONFIG"))
    app = FastAPI(title=PRODUCT_NAME, description=PRODUCT_TAGLINE, version="1.0.0")
    sessions: dict[str, ZeroZSession] = {}
    gen = None
    if cfg.use_model_generate and cfg.checkpoint:
        gen = build_generate_fn_from_checkpoint(cfg.checkpoint, device=cfg.device)

    @app.get("/health")
    def health() -> dict[str, Any]:
        bus = ToolBus(allowed=set(cfg.allowed_tools))
        return {
            "status": "ok",
            "product": PRODUCT_NAME,
            "tools": [s["name"] for s in bus.schemas()],
            "checkpoint": bool(cfg.checkpoint),
        }

    @app.get("/v1/policy")
    def policy() -> dict[str, str]:
        return {"product": PRODUCT_NAME, "policy": SYSTEM_PROMPT}

    @app.get("/v1/tools")
    def tools() -> dict[str, Any]:
        bus = ToolBus(allowed=set(cfg.allowed_tools))
        return {"tools": bus.schemas()}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        sid = req.session_id or "default"
        session = sessions.get(sid) or ZeroZSession(session_id=sid)
        sessions[sid] = session
        local_cfg = cfg
        if not req.use_model:
            local_cfg = ZeroZConfig(
                **{**cfg.__dict__, "use_model_generate": False}  # type: ignore
            )
        result = run_zeroz(
            req.message,
            cfg=local_cfg,
            session=session,
            generate_fn=gen if req.use_model else None,
        )
        return ChatResponse(
            answer=result.answer,
            stopped_reason=result.stopped_reason,
            steps=result.steps,
            metrics=dict(result.metrics()),
            session_id=session.session_id,
            tool_events=[
                {"name": e.name, "ok": e.ok, "detail": e.detail[:500], "ms": e.ms}
                for e in result.tool_events
            ],
        )

    return app


def main() -> None:
    import uvicorn

    host = os.getenv("ZEROZ_HOST", "0.0.0.0")
    port = int(os.getenv("ZEROZ_PORT", "8080"))
    uvicorn.run("sentinel.zeroz.server:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    main()
