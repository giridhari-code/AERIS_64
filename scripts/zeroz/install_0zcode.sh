#!/usr/bin/env bash
# Install 0zcode so typing 0zcode opens the agent
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT}/scripts/zeroz/0zcode"
SHORT="${ROOT}/scripts/zeroz/0z"

chmod +x "$BIN" "$SHORT" 2>/dev/null || true

mkdir -p "$HOME/.local/bin"
ln -sfn "$BIN" "$HOME/.local/bin/0zcode"
ln -sfn "$SHORT" "$HOME/.local/bin/0z"

SHELL_RC="$HOME/.bashrc"
if [[ "${SHELL:-}" == *zsh* ]]; then SHELL_RC="$HOME/.zshrc"; fi

if [[ -f "$SHELL_RC" ]] && grep -qF '.local/bin' "$SHELL_RC" 2>/dev/null; then
  echo "PATH already has ~/.local/bin in $SHELL_RC"
else
  echo "" >> "$SHELL_RC"
  echo "# 0zcode agent" >> "$SHELL_RC"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
  echo "Added PATH to $SHELL_RC"
fi

echo "Installed: ~/.local/bin/0zcode"
echo "Run:  source $SHELL_RC && 0zcode"
