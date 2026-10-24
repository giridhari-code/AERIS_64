# 0z code

**Rust implementation only** (Python agent removed).

```bash
cd apps/0zcode
cargo build --release
./target/release/0zcode
```

See [apps/0zcode/README.md](../../apps/0zcode/README.md).

## Official installer (Claude Code–style)

User-local install — **no sudo**.

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/giridhari-code/AERIS_64/main/scripts/zeroz/install.sh | bash

# Uninstall
curl -fsSL https://raw.githubusercontent.com/giridhari-code/AERIS_64/main/scripts/zeroz/uninstall.sh | bash
```

| Path | Purpose |
|------|---------|
| `~/.local/bin/0zcode` | launcher binary |
| `~/.0zcode/src` | cloned source |
| `~/.0zcode/install.meta` | install record |
| `~/.config/0zcode` | optional config |

Requires: `git`, `curl` (for rustup if needed), network on first run.
