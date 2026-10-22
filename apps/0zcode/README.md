# 0zcode

**Company-standard terminal agent** — Claude Code / OpenCode visual language · research & ops tools · **zero source code**.

| | |
|--|--|
| Binary | `0zcode` |
| Crate | `zeroz` |
| Version | 1.3.0 |
| UI | **std only** (raw mode via `stty`, no external crates) |
| Policy | No code write / edit / execute |
| LLM | Optional AERIS-64 link via `ZEROZ_AERIS_URL` |

## Architecture

```text
main → repl ──┬── term (raw mode, key decoder, widths)
              ├── editor (readline, history, slash menu)
              ├── tui (pure renderers + in-place Screen)
              └── agent / tools / policy / http
```

Tools / policy / http stay separate from the UI. Network work runs on a worker thread with a live spinner; Esc / Ctrl+C interrupts.

## Install

```bash
cd apps/0zcode
cargo build --release
cargo install --path .
```

## Usage

```text
0zcode                         # interactive TUI
0zcode "Weather in Mumbai?"
0zcode "write python code"     # policy refuse (exit 2)
0zcode tools
0zcode --version
```

### Interactive

A real terminal UI, still **Rust std only** (raw mode via `stty`, no crates):

```text
╭──────────────────────────────────────────────╮
│ ▣  Welcome to 0zcode  v1.3.0                 │
│    tools-first research agent · zero source  │
│ cwd    ~/work   model  AERIS-64 · tools-first│
╰──────────────────────────────────────────────╯
╭──────────────────────────────────────────────╮
│ ❯ 938+29                                     │
╰──────────────────────────────────────────────╯
● calc(938+29)  0.0s
  ⎿  938+29 = 967
```

* Rounded input box with line editing, wrapping and Devanagari-aware cursor
* `/` opens a command menu (↑ ↓ to pick, Tab to complete)
* Input history, saved across sessions in `~/.local/state/0zcode/history`
  (`ZEROZ_HISTORY_FILE` to override)
* One `● tool(query)` block per tool with a `⎿` result tree; long output
  collapses, `ctrl+o` / `/expand` shows all
* Live spinner while a network tool runs; `esc` or `ctrl+c` interrupts it and
  anything you type meanwhile is kept
* Bracketed paste, resize-aware redraw, terminal state always restored
* Piped / `TERM=dumb` / `ZEROZ_PLAIN=1` falls back to plain line mode (no escapes)

| Key | Action |
|-----|--------|
| `↑ ↓` | history · menu |
| `ctrl+a / e` | start / end of line |
| `alt+b / f` | word left / right |
| `ctrl+w` · `ctrl+u` · `ctrl+k` | delete word · to start · to end |
| `ctrl+o` | expand last tool output |
| `ctrl+l` | clear screen |
| `ctrl+c` | clear input · twice to exit |
| `ctrl+d` | exit on empty line |

Slash: `/help` `/tools` `/mcp` `/history` `/status` `/model` `/version`
`/copy` `/expand` `/clear` `/exit`

### Network permission

Weather, search, wiki, FX, github, crypto prompts are network tools. In TUI they run with spinner; Esc interrupts.

### Config

`~/.config/0zcode/config.toml` or `ZEROZ_CONFIG`

```toml
model_label = "AERIS-64 · tools-first"
aeris_url = "http://127.0.0.1:8000/chat"  # AERIS-64 / SENTINEL LLM fallback
offline = false
color = true
http_timeout_secs = 12
```

| Env | Role |
|-----|------|
| `ZEROZ_HTTP_TIMEOUT_SECS` | HTTP timeout 1–60 (default 12) |
| `ZEROZ_MAX_OUTPUT_CHARS` | Tool output clamp (default 8000) |
| `ZEROZ_PLAIN` | `1` forces plain line mode (no TUI) |
| `ZEROZ_HISTORY_FILE` | Input history file path |
| `ZEROZ_MODEL_LABEL` | TUI display label only |
| `ZEROZ_AERIS_URL` | AERIS-64 / SENTINEL chat endpoint |
| `ZEROZ_USER_AGENT` | HTTP User-Agent override |
| `ZEROZ_NO_COLOR` / `NO_COLOR` | disable ANSI |
| `ZEROZ_OFFLINE` | calc + commands only |

### AERIS-64 link

When tools cannot answer and `aeris_url` / `ZEROZ_AERIS_URL` is set, the query is forwarded to your local AERIS-64 / SENTINEL endpoint. Coding requests are still refused by policy before any LLM call.

```bash
# start model server
uvicorn "sentinel.api.app:create_app" --factory --port 8000
export ZEROZ_AERIS_URL=http://127.0.0.1:8000/chat
0zcode
```

## Test

```bash
cargo test
cargo test -- --nocapture
cargo clippy --all-targets
```

## CI

`.github/workflows/0zcode.yml` — fmt, clippy, test, release build.

## Non-goals

- Source code generation or shell execution
- Fake LLM token streaming (tools-first progress only)
- External TUI crates (ratatui / crossterm) — this binary is std-only
