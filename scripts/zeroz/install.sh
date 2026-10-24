#!/usr/bin/env bash
# 0zcode installer — company-standard (Claude Code–style UX)
# Installs into $HOME; does not need root.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/giridhari-code/AERIS_64/main/scripts/zeroz/install.sh | bash
#   curl -fsSL …/install.sh | bash -s latest
#   bash install.sh [latest|stable|path]
#
# Env:
#   ZEROZ_INSTALL_DIR   default: $HOME/.0zcode
#   ZEROZ_BIN_DIR       default: $HOME/.local/bin
#   ZEROZ_REPO_URL      default: https://github.com/giridhari-code/AERIS_64.git
#   ZEROZ_INSTALL_ALLOW_SUDO=1  allow intentional root install

set -euo pipefail

TARGET="${1:-latest}"
REPO_URL="${ZEROZ_REPO_URL:-https://github.com/giridhari-code/AERIS_64.git}"
INSTALL_DIR="${ZEROZ_INSTALL_DIR:-$HOME/.0zcode}"
BIN_DIR="${ZEROZ_BIN_DIR:-$HOME/.local/bin}"
SRC_DIR="$INSTALL_DIR/src"
BIN_NAME="0zcode"

# ── refuse sudo-from-user (same rationale as Claude installer) ─────────────
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ] \
  && [ -z "${ZEROZ_INSTALL_ALLOW_SUDO:-}" ]; then
  echo "Error: do not run this installer with sudo." >&2
  echo "" >&2
  echo "0zcode installs into your home directory and does not need root." >&2
  echo "With sudo, files would land under root's home and '0zcode' would not" >&2
  echo "be on your normal user PATH." >&2
  echo "" >&2
  echo "Re-run without sudo, e.g.:" >&2
  echo "  curl -fsSL https://raw.githubusercontent.com/giridhari-code/AERIS_64/main/scripts/zeroz/install.sh | bash" >&2
  echo "" >&2
  echo "To install for the root user intentionally:" >&2
  echo "  curl -fsSL …/install.sh | sudo ZEROZ_INSTALL_ALLOW_SUDO=1 bash" >&2
  exit 1
fi

if [[ ! "$TARGET" =~ ^(stable|latest|path)$ ]] && [[ ! "$TARGET" =~ ^[0-9]+\.[0-9]+ ]]; then
  # allow bare "latest" default; reject garbage
  if [ -n "$1" ] && [[ ! "$1" =~ ^(stable|latest|path)$ ]]; then
    echo "Usage: $0 [latest|stable|path]" >&2
    exit 1
  fi
fi

echo "→ 0zcode installer"
echo "  install dir: $INSTALL_DIR"
echo "  bin dir:     $BIN_DIR"
echo "  target:      $TARGET"
echo ""

# ── tools ──────────────────────────────────────────────────────────────────
need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but not found." >&2
    return 1
  fi
}

ensure_rust() {
  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    local ver
    ver="$(rustc --version 2>/dev/null || true)"
    echo "→ Rust found: $ver"
    return 0
  fi
  echo "→ Rust not found — installing via rustup (user-local)…"
  if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required to install rustup." >&2
    exit 1
  fi
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    . "$HOME/.cargo/env"
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo still not on PATH after rustup." >&2
    exit 1
  fi
  echo "→ Rust installed: $(rustc --version)"
}

# ── platform note (build-from-source works broadly) ────────────────────────
case "$(uname -s)" in
  Darwin|Linux) ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "Windows: use WSL or build manually with cargo." >&2
    exit 1
    ;;
  *)
    echo "Unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64|amd64|arm64|aarch64) ;;
  *)
    echo "Warning: architecture $(uname -m) is not CI-tested; continuing." >&2
    ;;
esac

need_cmd git || { echo "Install git, then re-run." >&2; exit 1; }
ensure_rust

mkdir -p "$INSTALL_DIR" "$BIN_DIR" "$SRC_DIR"

# ── fetch source ───────────────────────────────────────────────────────────
if [ -d "$SRC_DIR/.git" ]; then
  echo "→ Updating repo in $SRC_DIR …"
  git -C "$SRC_DIR" fetch --depth 1 origin main 2>/dev/null \
    || git -C "$SRC_DIR" fetch --depth 1 origin master 2>/dev/null \
    || true
  git -C "$SRC_DIR" checkout main 2>/dev/null \
    || git -C "$SRC_DIR" checkout master 2>/dev/null \
    || true
  git -C "$SRC_DIR" pull --ff-only 2>/dev/null || true
else
  echo "→ Cloning $REPO_URL → $SRC_DIR …"
  rm -rf "$SRC_DIR"
  git clone --depth 1 --branch main "$REPO_URL" "$SRC_DIR" 2>/dev/null \
    || git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

APP_DIR="$SRC_DIR/apps/0zcode"
if [ ! -f "$APP_DIR/Cargo.toml" ]; then
  echo "Error: apps/0zcode not found in repo (unexpected layout)." >&2
  exit 1
fi

# ── build + install binary ─────────────────────────────────────────────────
echo "→ Building release binary (this may take a few minutes)…"
(
  cd "$APP_DIR"
  # Prefer installing into cargo bin, then link into ~/.local/bin for Claude-like path
  cargo build --release
)

BUILT="$APP_DIR/target/release/$BIN_NAME"
if [ ! -x "$BUILT" ]; then
  echo "Error: build did not produce $BUILT" >&2
  exit 1
fi

# Also cargo install for consistent updates
echo "→ Installing to Cargo bin…"
cargo install --path "$APP_DIR" --force

CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin/$BIN_NAME"
if [ -x "$CARGO_BIN" ]; then
  ln -sfn "$CARGO_BIN" "$BIN_DIR/$BIN_NAME"
else
  install -m 755 "$BUILT" "$BIN_DIR/$BIN_NAME"
fi

# Record install metadata
{
  echo "installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO_URL"
  echo "src=$SRC_DIR"
  echo "bin=$BIN_DIR/$BIN_NAME"
  if command -v "$BIN_DIR/$BIN_NAME" >/dev/null 2>&1; then
    echo "version=$("$BIN_DIR/$BIN_NAME" --version 2>/dev/null || echo unknown)"
  fi
} >"$INSTALL_DIR/install.meta"

# ── PATH hint ──────────────────────────────────────────────────────────────
path_ok=false
case ":$PATH:" in
  *":$BIN_DIR:"*) path_ok=true ;;
esac

echo ""
if command -v "$BIN_NAME" >/dev/null 2>&1 || [ -x "$BIN_DIR/$BIN_NAME" ]; then
  VER="$("$BIN_DIR/$BIN_NAME" --version 2>/dev/null || echo 0zcode)"
  echo "✅ Installation complete — $VER"
else
  echo "✅ Binary installed at $BIN_DIR/$BIN_NAME"
fi
echo ""
echo "  Binary:  $BIN_DIR/$BIN_NAME"
echo "  Source:  $SRC_DIR"
echo "  Meta:    $INSTALL_DIR/install.meta"
echo ""

if [ "$path_ok" = false ]; then
  echo "Add to PATH (run once):"
  echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
  echo "  # fish: fish_add_path \$HOME/.local/bin"
  echo "  # zsh:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
  echo ""
fi

echo "Run:"
echo "  $BIN_NAME"
echo ""
echo "Uninstall:"
echo "  curl -fsSL https://raw.githubusercontent.com/giridhari-code/AERIS_64/main/scripts/zeroz/uninstall.sh | bash"
echo "  # or: bash $SRC_DIR/scripts/zeroz/uninstall.sh"
echo ""
