#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serve}"

case "$MODE" in
  serve)
    # create_app reads NEUROFIELD_CHECKPOINT / NEUROFIELD_CONFIG / NEUROFIELD_DEVICE itself.
    # (The old "module:create_app(checkpoint=...)" call-expression is NOT valid uvicorn syntax.)
    export NEUROFIELD_CHECKPOINT="${NEUROFIELD_CHECKPOINT:-/app/checkpoints/model}"
    export NEUROFIELD_DEVICE="${NEUROFIELD_DEVICE:-cuda}"
    HOST="${NEUROFIELD_HOST:-0.0.0.0}"
    PORT="${NEUROFIELD_PORT:-8000}"
    WORKERS="${NEUROFIELD_WORKERS:-1}"

    if [ "$WORKERS" != "1" ]; then
      echo "WARNING: $WORKERS workers = $WORKERS copies of the model in memory and per-process" >&2
      echo "         sessions/rate-limits. Prefer 1 worker + more replicas with sticky sessions." >&2
    fi
    if [ ! -e "$NEUROFIELD_CHECKPOINT" ]; then
      echo "ERROR: checkpoint not found: $NEUROFIELD_CHECKPOINT" >&2
      exit 2
    fi

    echo "Starting NeuroField inference server"
    echo "  checkpoint: $NEUROFIELD_CHECKPOINT"
    echo "  device:     $NEUROFIELD_DEVICE"
    echo "  listen:     $HOST:$PORT"

    exec python -m uvicorn "neurofield.serving.server:create_app" --factory \
      --host "$HOST" --port "$PORT" --workers "$WORKERS" \
      --log-level info --no-access-log
    ;;

  train)
    exec neurofield-train "${@:2}"
    ;;

  eval)
    exec neurofield-eval "${@:2}"
    ;;

  bash)
    exec /bin/bash
    ;;

  *)
    echo "Unknown mode: $MODE (serve|train|eval|bash)"
    exit 1
    ;;
esac
