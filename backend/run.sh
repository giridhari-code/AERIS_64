#!/usr/bin/env bash
# Frontier backend entry — inference API only
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:packages"
exec uvicorn "sentinel.api.app:create_app" --factory --host 0.0.0.0 --port "${PORT:-8000}" "$@"
