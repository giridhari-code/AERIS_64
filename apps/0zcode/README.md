# 0zcode

**Company-standard terminal agent** — Claude Code / OpenCode visual language · research & ops tools · **zero source code**.

| | |
|--|--|
| Binary | `0zcode` |
| Crate | `zeroz` |
| Version | 1.3.1 |
| UI | **std only** (raw mode via `stty`, no external crates) |
| Policy | No code write / edit / execute |
| LLM | AERIS-64 chat + live streaming via `ZEROZ_AERIS_URL` |

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

On launch, **0zcode auto-starts AERIS-64** at `http://127.0.0.1:8000` when it is down (`ZEROZ_AERIS_AUTO=0` to disable; set `ZEROZ_REPO` / `NEUROFIELD_CHECKPOINT` if needed).


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
 ███████   0zcode v1.3.0
██ ███ ██  AERIS-64 · tools-first
  ▀▀ ▀▀    ~/work

❯ 938+29                                     ← grey band (echo)

● 938+29 = 967

────────────────────────────────────────────
❯ █
────────────────────────────────────────────
  ? for shortcuts            AERIS-64 · tools-first
```

* Claude Code–style prompt between two full-width rules, with line editing, wrapping and Devanagari-aware cursor
* Submitted line stays in the scrollback as a grey band; replies start with `●`
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
| `?` (empty prompt) | shortcuts overlay |
| `ctrl+o` | expand last tool output (scrollable: `↑ ↓` `j k` `PgUp PgDn`) |
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
| `ZEROZ_MOUSE` | `0` turns off wheel / touch scrolling of the transcript (then a swipe on the alt screen sends ↑ ↓) |
| `ZEROZ_ALT_SCREEN` | `0` stays on the main screen (chat remains in scrollback after exit) |
| `ZEROZ_HISTORY_FILE` | Input history file path |
| `ZEROZ_MODEL_LABEL` | TUI display label only |
| `ZEROZ_AERIS_URL` | AERIS-64 server, e.g. `http://127.0.0.1:8000` |
| `ZEROZ_AERIS_KEY` | `X-API-Key` when the server uses `NEUROFIELD_API_KEYS` |
| `ZEROZ_AERIS_STREAM` | `0` = one reply instead of live tokens (default on) |
| `ZEROZ_AERIS_MAX_TOKENS` | `max_new_tokens` per request (default 256, max 8192) |
| `ZEROZ_AERIS_TIMEOUT_SECS` | give up after this long without data (default 60) |
| `ZEROZ_USER_AGENT` | HTTP User-Agent override |
| `ZEROZ_NO_COLOR` / `NO_COLOR` | disable ANSI |
| `ZEROZ_OFFLINE` | calc + commands only |

### AERIS-64 link (real LLM)

Anything no tool can answer — chat, open questions, Hinglish — goes to your AERIS-64 / SENTINEL server. Tools still win when one matches (`938+29`, `weather in Mumbai`), and coding requests are refused by policy *before* any model call.

```bash
scripts/zeroz/run_aeris.sh        # starts the server, waits for the model, runs 0zcode, stops the server
# or by hand:
NEUROFIELD_CHECKPOINT=checkpoints/local_run PYTHONPATH=packages \
  uvicorn "sentinel.api.app:create_app" --factory --port 8000
ZEROZ_AERIS_URL=http://127.0.0.1:8000 0zcode
```

* Live tokens over `POST /v1/chat/stream` (SSE), word-wrapped under `●`; `esc` drops the connection. `ZEROZ_AERIS_STREAM=0` uses `POST /v1/chat` (one polished reply) instead.
* One `session_id` per run, so the server keeps the conversation; `/clear` starts a new one.
* `/status` and `/model` ping `GET /v1/models` and show `online · <model>` or the exact problem (not running, 401 → set `ZEROZ_AERIS_KEY`, 503 → model not loaded).
* Only `http://` (this binary is std-only, no TLS). Any path in the URL is ignored, so old `…:8000/chat` values keep working.
* With the model linked, a `?` question that matches no tool goes to the model instead of a web search.
* The model does not see tool results yet; it answers from its own weights and session history.
* `ZEROZ_OFFLINE=1` disables the link.

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
- Fake streaming — tokens shown are the server's real token stream
- External TUI crates (ratatui / crossterm) — this binary is std-only
