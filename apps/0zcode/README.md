# 0z code (Rust)

**Company-standard terminal agent** — research & ops tools, **zero source code**.

| | |
|--|--|
| Binary | `0zcode` |
| Version | 1.0.0 |
| Dependencies | **none** (Rust std only) |
| Policy | No code write / execute |

## Build

```bash
cd apps/0zcode
cargo build --release
cargo test
./target/release/0zcode
```

## Install

```bash
cargo install --path .
# ensures `0zcode` on PATH (cargo bin)
```

## Usage

```text
0zcode                         # interactive
0zcode "Weather in Mumbai?"
0zcode "10 USD to INR"
0zcode "write python code"     # policy refuse (exit 2)
0zcode tools
0zcode --version
```

### Interactive

```text
0z code v1.0.0 — Agentic work. Zero code.  |  ready

0z> Weather in Delhi?
┌ weather ──────────────────────────
│ …
└──────────────────────────────────

0z> /tools
```

## Configuration (env)

| Variable | Meaning |
|----------|---------|
| `ZEROZ_HTTP_TIMEOUT_SECS` | HTTP timeout (default 12) |
| `ZEROZ_NO_COLOR` / `NO_COLOR` | disable ANSI colors |

## Architecture

```text
main → agent::handle
         ├─ policy (code refuse)
         ├─ route (deterministic)
         └─ tools (HTTP JSON APIs)
```

Modules: `policy`, `agent`, `tools`, `http`, `config`, `error`, `tui`.

## License

MIT — see repository root.
