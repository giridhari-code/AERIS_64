#!/usr/bin/env bash
# Install Rust 0zcode binary as `0zcode`
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/apps/0zcode"
cargo build --release
mkdir -p "$HOME/.local/bin"
ln -sfn "$ROOT/apps/0zcode/target/release/0zcode" "$HOME/.local/bin/0zcode"
echo "Installed: ~/.local/bin/0zcode"
echo "Run: export PATH=\"\$HOME/.local/bin:\$PATH\" && 0zcode"
