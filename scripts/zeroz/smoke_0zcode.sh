#!/usr/bin/env bash
# CI smoke for 0zcode — offline calc must work without network.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/apps/0zcode"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/0zcode-target}"
cargo build -q
BIN="$CARGO_TARGET_DIR/debug/0zcode"
export ZEROZ_OFFLINE=1
export ZEROZ_NO_COLOR=1

out="$("$BIN" "2+2")"
echo "$out" | grep -q "4" || { echo "FAIL: calc 2+2 => $out"; exit 1; }

out2="$("$BIN" "calculate 25*4")"
echo "$out2" | grep -q "100" || { echo "FAIL: 25*4 => $out2"; exit 1; }

code=0
"$BIN" "write python code" >/tmp/0z_policy.txt 2>&1 || code=$?
if [[ "$code" != "2" ]]; then
  echo "FAIL: expected exit 2 on code request, got $code"
  exit 1
fi

echo "0zcode smoke OK"
