#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
echo "health vs $BASE"
curl -sf "$BASE/readyz" 2>/dev/null || curl -sf "$BASE/health" 2>/dev/null || curl -sf "$BASE/" >/dev/null
echo "ready ok"
curl -sf "$BASE/v1/tools/mcp" 2>/dev/null | head -c 200 || echo "(mcp may require auth)"
echo
