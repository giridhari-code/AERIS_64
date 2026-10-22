# 0zcode

**Company-standard terminal agent** — research & ops tools, **zero source code**.

| | |
|--|--|
| Binary | `0zcode` |
| Crate | `zeroz` |
| Version | 1.0.0 |
| Dependencies | **Rust std only** (no crates.io deps) |
| Policy | No code write / edit / execute |

## Build & test

```bash
cd apps/0zcode
cargo test
cargo build --release
./target/release/0zcode --version
```

## Install

```bash
cargo install --path .
```

## Usage

```text
0zcode                         # interactive TUI
0zcode "Weather in Mumbai?"
0zcode "10 USD to INR"
0zcode "write python code"     # policy refuse (exit 2)
0zcode tools
0zcode --version
```

### Interactive

```text
 ╭───╮
 │ ◉ ◉ │
 0zcode  v1.0.0
 …

❯ weather in Delhi
┌ weather ──────────────────────────
│ …
└──────────────────────────────────
```

Slash: `/help` `/tools` `/status` `/version` `/clear` `/quit`

## Configuration (env)

| Variable | Meaning |
|----------|---------|
| `ZEROZ_HTTP_TIMEOUT_SECS` | HTTP timeout 1–60 (default 12) |
| `ZEROZ_MAX_OUTPUT_CHARS` | Tool output clamp (default 8000) |
| `ZEROZ_MODEL_LABEL` | TUI display label only |
| `ZEROZ_USER_AGENT` | HTTP User-Agent override |
| `ZEROZ_NO_COLOR` / `NO_COLOR` | disable ANSI |
| `ZEROZ_COLOR=1` | force ANSI on |

## Architecture

```text
main → agent::handle
         ├─ policy (code refuse)
         ├─ slash meta (/help, /tools, …)
         ├─ route (deterministic keywords)
         └─ tools (HTTP JSON APIs, std only)
tui  → ANSI chrome (logo, boxes, status)
```

## Brand

- `assets/logo.svg` — app icon  
- `assets/logo-mark.txt` — plain-text mark  
- TUI robot glyph in `src/tui.rs`

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 2 | policy refusal (coding request) |

## License

MIT — see repository root.
