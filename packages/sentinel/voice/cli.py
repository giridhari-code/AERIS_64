"""CLI entrypoints for Indo-Voice-3."""

from __future__ import annotations

import argparse
import logging
import sys

from sentinel.voice_settings import get_settings
from sentinel.voice.factory import build_voice_system


def cmd_serve(args):
    import uvicorn
    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    uvicorn.run(
        "sentinel.voice.server:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )


def cmd_generate(args):
    settings = get_settings()
    adapter, model, tok, stt, tts, settings = build_voice_system(settings, checkpoint=args.ckpt)
    if tok is None or model is None:
        print("ERROR: checkpoint/tokenizer required", file=sys.stderr)
        sys.exit(1)
    session = adapter.create_session("cli", language=settings.primary_language)
    ids = tok.encode(args.text, return_tensors="pt").to(settings.device)
    print(args.text)
    print("---")
    for tid in adapter.generate_stream(ids, session, max_new_tokens=args.max_new or settings.max_new_tokens):
        sys.stdout.write(tok.decode([tid]))
        sys.stdout.flush()
    print()


def cmd_health(args):
    settings = get_settings()
    from pathlib import Path
    ckpt = Path(args.ckpt or settings.checkpoint_dir)
    print("checkpoint:", ckpt, "exists=" + str(ckpt.exists()))
    print("device:", settings.device)
    print("stt:", settings.stt_provider)
    print("tts:", settings.tts_provider)
    if ckpt.exists():
        adapter, model, tok, stt, tts, _ = build_voice_system(settings, checkpoint=str(ckpt))
        print("model_params:", sum(p.numel() for p in model.parameters()) if model else None)
        print("stt_backend:", type(stt).__name__)
        print("tts_backend:", type(tts).__name__)
        print("status: ok")
    else:
        print("status: missing_checkpoint")
        sys.exit(1)


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(prog="indo-voice", description="Indo-Voice-3")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Run HTTP/WebSocket server")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    g = sub.add_parser("generate", help="Text generation against checkpoint")
    g.add_argument("--text", required=True)
    g.add_argument("--ckpt", default=None)
    g.add_argument("--max-new", type=int, default=None)
    g.set_defaults(func=cmd_generate)

    h = sub.add_parser("health", help="Validate checkpoint and backends")
    h.add_argument("--ckpt", default=None)
    h.set_defaults(func=cmd_health)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
