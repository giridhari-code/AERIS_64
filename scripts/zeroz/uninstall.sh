#!/usr/bin/env bash
# 0zcode uninstaller — removes user-local install (Claude Code–style)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/giridhari-code/AERIS_64/main/scripts/zeroz/uninstall.sh | bash
#   bash uninstall.sh
#   bash uninstall.sh --keep-src    # leave ~/.0zcode/src clone
#
# Env:
#   ZEROZ_INSTALL_DIR  default $HOME/.0zcode
#   ZEROZ_BIN_DIR      default $HOME/.local/bin

set -euo pipefail

KEEP_SRC=false
for arg in "$@"; do
  case "$arg" in
    --keep-src) KEEP_SRC=true ;;
    -h|--help)
      echo "Usage: $0 [--keep-src]"
      exit 0
      ;;
  esac
done

INSTALL_DIR="${ZEROZ_INSTALL_DIR:-$HOME/.0zcode}"
BIN_DIR="${ZEROZ_BIN_DIR:-$HOME/.local/bin}"
BIN_NAME="0zcode"
CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin/$BIN_NAME"

echo "→ 0zcode uninstall"
echo "  bin dir:     $BIN_DIR"
echo "  install dir: $INSTALL_DIR"
echo ""

# Binary links
rm -f "$BIN_DIR/$BIN_NAME"
echo "  removed $BIN_DIR/$BIN_NAME (if present)"

# cargo install copy
if command -v cargo >/dev/null 2>&1; then
  cargo uninstall zeroz 2>/dev/null && echo "  cargo uninstall zeroz — ok" || true
fi
rm -f "$CARGO_BIN"
echo "  removed $CARGO_BIN (if present)"

# which leftovers
if command -v "$BIN_NAME" >/dev/null 2>&1; then
  EXTRA="$(command -v "$BIN_NAME" || true)"
  if [ -n "$EXTRA" ] && [ -f "$EXTRA" ]; then
    rm -f "$EXTRA"
    echo "  removed $EXTRA"
  fi
fi

# Config / state (optional user data — remove by default for clean uninstall)
for d in \
  "$HOME/.config/0zcode" \
  "$HOME/.local/state/0zcode" \
  "$HOME/.local/share/0zcode"
do
  if [ -e "$d" ]; then
    rm -rf "$d"
    echo "  removed $d"
  fi
done

if [ "$KEEP_SRC" = true ]; then
  echo "  keeping source at $INSTALL_DIR/src (--keep-src)"
  rm -f "$INSTALL_DIR/install.meta"
else
  if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "  removed $INSTALL_DIR"
  fi
fi

echo ""
if command -v "$BIN_NAME" >/dev/null 2>&1; then
  echo "⚠  '$BIN_NAME' still on PATH at: $(command -v "$BIN_NAME")"
  echo "   Remove it manually if it is a different install."
else
  echo "✅ 0zcode uninstalled"
fi
echo ""
