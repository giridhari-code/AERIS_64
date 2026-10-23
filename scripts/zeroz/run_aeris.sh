#!/usr/bin/env bash
# One command: start the AERIS-64 server (if it is not already running),
# wait until the model is loaded, run 0zcode linked to it, stop the server on exit.
#
#   scripts/zeroz/run_aeris.sh              # interactive TUI
#   scripts/zeroz/run_aeris.sh "namaste"    # one-shot
#
# Env: NEUROFIELD_CHECKPOINT (folder with model.safetensors), PORT (default 8000),
#      PYTHON (default python3), ZEROZ_AERIS_KEY (if the server uses API keys).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PORT:-8000}"
PY="${PYTHON:-python3}"
URL="http://127.0.0.1:${PORT}"

CKPT="${NEUROFIELD_CHECKPOINT:-}"
if [ -z "$CKPT" ]; then
  for d in checkpoints/local_run checkpoints/examples/AERIS_main; do
    if [ -f "$ROOT/$d/model.safetensors" ]; then CKPT="$ROOT/$d"; break; fi
  done
fi

SRV=""
cleanup() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if curl -fsS "$URL/readyz" >/dev/null 2>&1; then
  echo "• AERIS-64 already running at $URL" >&2
else
  if [ -z "$CKPT" ]; then
    echo "No checkpoint found. Train one or set NEUROFIELD_CHECKPOINT=/path/to/folder" >&2
    echo "(folder must contain model.safetensors — weights are not stored in git)" >&2
    exit 1
  fi
  echo "• starting AERIS-64 ($CKPT) on :$PORT" >&2
  NEUROFIELD_CHECKPOINT="$CKPT" PYTHONPATH="$ROOT/packages${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -m uvicorn "sentinel.api.app:create_app" --factory --host 127.0.0.1 --port "$PORT" \
    --log-level warning >"${TMPDIR:-/tmp}/aeris-server.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 120); do
    curl -fsS "$URL/readyz" >/dev/null 2>&1 && break
    kill -0 "$SRV" 2>/dev/null || { echo "server exited — see ${TMPDIR:-/tmp}/aeris-server.log" >&2; exit 1; }
    sleep 0.5
  done
  curl -fsS "$URL/readyz" >/dev/null 2>&1 || { echo "server not ready — see ${TMPDIR:-/tmp}/aeris-server.log" >&2; exit 1; }
fi

export ZEROZ_AERIS_URL="$URL"
cargo run --release --quiet --manifest-path "$ROOT/apps/0zcode/Cargo.toml" -- "$@"
