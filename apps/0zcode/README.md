# 0zcode

**Company-standard terminal agent** — research & ops tools, **zero source code**.

| | |
|--|--|
| Binary | `0zcode` |
| Crate | `zeroz` |
| Version | 1.2.0 |
| UI | ratatui + crossterm + tokio |
| Policy | No code write / edit / execute |

## Architecture

```text
UI (ratatui)  --UiEvent-->  agent worker (spawn_blocking)
     ^                         |
     +-- AgentUpdate (gen id) -+

Tools / policy / http stay separate from UI.
Cancel bumps request_gen; stale updates are dropped.
```

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

### Keybindings

| Key | Action |
|-----|--------|
| Enter | send line |
| Esc | deny permission → cancel in-flight → quit |
| Ctrl+C | quit |
| Tab | complete `/` command or `@` path |
| y / a / n | network allow once / always / deny |

### Slash commands

`/help` `/clear` `/model` `/tools` `/mcp` `/status` `/compact` `/resume` `/config` `/quit`

### Network permission

Weather, search, wiki, FX, github, crypto prompts: **y** / **a** / **n**.

### Sessions

Saved under `~/.local/share/0zcode/sessions/last.jsonl`  
`/resume` reloads · `/compact` keeps last 40 messages.

### Theme

`blood and ash` (default).  
`ZEROZ_THEME=light` · `ZEROZ_COLOR=16` · `NO_COLOR=1`

### Config

`~/.config/0zcode/config.toml` or `ZEROZ_CONFIG`

```toml
model_label = "local / tools-first"
offline = false
color = true
http_timeout_secs = 12
```

## Test

```bash
cargo test
cargo test -- --nocapture   # snapshot buffer dumps
cargo clippy --all-targets
```

## CI

`.github/workflows/0zcode.yml` — fmt, clippy, test, release build.

## Non-goals

- Source code generation or shell execution
- Fake LLM token streaming (tools-first progress only)
